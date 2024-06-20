"""Streamers for core PyTorch modules"""

import abc
import typing as T

import torch as pt
import torch.nn as nn
import torch.nn.functional as ptf


class BaseStream(nn.Module, abc.ABC):
    """Base class for all streamers"""

    @abc.abstractmethod
    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        """Processes an input tensor through the stream

        Args:
            x: input tensor, shaped `[C, T]`
            final: true if this is the final input, used to flush internal buffers

        Returns:
            y: streaming output, if enough inputs have been buffered, null otherwise

        """

        raise NotImplementedError()


class Sequential1dStream(BaseStream):
    """Streamer for `torch.nn.Sequential` networks over 1D `[B, C, T]` tensors

    Args:
        net: sequential network to stream

    Example:
        ```python
        net = nn.Sequential(
           nn.Conv1d(1, 128, 1),
           nn.Conv1d(128, 1, 1),
        )
        stream = pts.Sequential1dStream(net)
        ```

    """

    def __init__(self, net: nn.Sequential):
        super().__init__()

        self.streams = nn.ModuleList([create_stream(m) for m in net.children()])

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        y: pt.Tensor | None = x
        for i, stream in enumerate(self.streams):
            if y is not None:
                y = stream(y, final=final)
        return y


class Residual1dStream(BaseStream):
    """Streamer for ResNets

    This streamer streams over the inner layers of a residual network (the
    part without the residual connection). It buffers the inputs so that they
    can be added (without padding) to the outputs after processing through the
    network's receptive field.

    Args:
        net: inner part of the residual network to stream

    Example:
        ```python
        net = pts.Residual1d(
            nn.Conv1d(1, 128, 1),
            nn.Conv1d(128, 1, 1),
        )
        stream = pts.Residual1dStream(net)
        ```

    """

    x_buffer: list[pt.Tensor]

    def __init__(self, net: nn.Module):
        super().__init__()

        # calculate the size of the receptive field of the resnet,
        # which determines the amount of padding used by the net
        convs = [m for m in net.modules() if isinstance(m, nn.Conv1d)]
        field = (
            sum(
                convs[i].dilation[0] * (convs[i].kernel_size[0] - 1)
                for i in range(len(convs))
            )
            + 1
        )
        self.padding = (field - 1) // 2

        # create streams for the inner network
        self.stream = create_stream(net.net if isinstance(net, Residual1d) else net)

        self.x_buffer = []
        self.x_offset = 0

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        # save off the inputs for the residual connectin and
        # process the inner streams
        self.x_buffer.append(x)
        y = self.stream(x, final=final)

        if y is not None:
            # stack any buffered residuals
            x = pt.cat(self.x_buffer, dim=-1)
            self.x_buffer.clear()

            # skip over any left padding in the residuals
            offset = max(min(self.padding - self.x_offset, x.shape[-1]), 0)
            if offset > 0:
                x = x[..., offset:]
                self.x_offset = offset

            # add the residuals to the output and buffer any remaining residuals
            length = y.shape[-1]
            y += x[..., :length]
            self.x_buffer.append(x[..., length:])

        return y


class Residual1d(nn.Module):
    """PyTorch module for wrapping a network with a residual connection

    Note this is not a streamer, just a utility module to make it easier to
    build streamable residual networks.

    Args:
        net: inner network, which will be wrapped with a residual connection

    Raises:
        ValueError: if the inner network contains strided convolutions,
            which are not supported

    Example:
        ```python
        net = nn.Sequential(
            nn.Conv1d(1, 128, 1),
            pts.Residual1d(
                nn.GELU(),
                nn.Conv1d(128, 256, 1),
                nn.GELU(),
                nn.Conv1d(256, 128, 1),
            ),
            nn.Conv1d(128, 1, 1),
        )
        ```

    """

    def __init__(self, net: nn.Module):
        super().__init__()

        if any(m.stride[0] != 1 for m in net.modules() if isinstance(m, nn.Conv1d)):
            raise ValueError("invalid_residual_stride")

        self.net = net

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        y = self.net(x)

        # skip over padding in the residuals, for non-padding resnets
        p = x.shape[-1] - y.shape[-1]
        if p > 0:
            x = x[..., p // 2 : -(p // 2 + p % 2)]

        return y + x


class Conv1dStream(BaseStream):
    """Streamer for Conv1d modules

    Note that the stream ignores any padding configuration for the
    convolution, performing an exact (valid) convolution instead.

    Args:
        net: Conv1d layer to stream

    """

    x_buffer: list[pt.Tensor]

    def __init__(self, net: nn.Conv1d):
        super().__init__()

        self.weight = nn.Parameter(
            pt._weight_norm(net.weight_v, net.weight_g)
            if hasattr(net, "weight_g")
            else net.weight
        )
        self.bias = nn.Parameter(net.bias) if net.bias is not None else None

        self.field = net.dilation[0] * (net.kernel_size[0] - 1) + 1
        self.stride = net.stride[0]
        self.dilation = net.dilation[0]
        self.groups = net.groups

        self.x_buffer = []
        self.x_length = 0
        self.x_skip = 0

    def __repr__(self) -> str:
        o, i, k = self.weight.shape
        return f"Conv1dStream({i}, {o}, kernel_size={k}, stride={self.stride})"

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        y: pt.Tensor | None = None

        # buffer inputs until we fill the receptive field
        self.x_buffer.append(x)
        self.x_length += x.shape[-1]
        if self.x_length - self.x_skip >= self.field:
            # stack the inputs for convolutional processing
            x = pt.cat(self.x_buffer, dim=-1)[..., self.x_skip :]
            self.x_buffer.clear()
            self.x_length = 0

            # calculate sizes of the inputs/outputs
            steps = (x.shape[-1] - self.field) // self.stride
            valid = self.field + steps * self.stride
            next_ = (steps + 1) * self.stride

            # apply the convolution over the available inputs
            with pt.no_grad():
                y = ptf.conv1d(
                    x[..., :valid].unsqueeze(0),
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                ).squeeze(0)

            # save off any inputs needed for the next application
            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)
            self.x_skip = max(next_ - x.shape[-1], 0)

        return y


class ConvTranspose1dStream(BaseStream):
    """Streamer for ConvTranspose1d modules

    Note that the stream ignores any padding configuration for the
    convolution, performing an exact (valid) transposed convolution instead.

    Args:
        net: ConvTranspose1d layer to stream

    """

    y_buffer: pt.Tensor | None

    def __init__(self, net: nn.ConvTranspose1d):
        super().__init__()

        self.weight = nn.Parameter(
            pt._weight_norm(net.weight_v, net.weight_g)
            if hasattr(net, "weight_g")
            else net.weight
        )
        self.bias = (
            nn.Parameter(net.bias.unsqueeze(-1)) if net.bias is not None else None
        )

        self.field = net.dilation[0] * (net.kernel_size[0] - 1) + 1
        self.stride = net.stride[0]
        self.dilation = net.dilation[0]
        self.groups = net.groups
        self.padding = max(self.stride - self.field, 0)
        self.overlap = self.field - self.stride

        self.y_buffer = None
        self.y_offset = 0

    def __repr__(self) -> str:
        i, o, k = self.weight.shape
        return f"ConvTranspose1dStream({i}, {o}, kernel_size={k}, stride={self.stride})"

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        y: pt.Tensor | None = None

        # apply the transposed convolution
        # no input buffering is needed, since the convolution is always
        # applied over each input element
        with pt.no_grad():
            y = ptf.conv_transpose1d(
                x.unsqueeze(0),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            ).squeeze(0)

        # apply output padding to initialize overlap
        y = ptf.pad(y, [0, self.padding])

        # accumulate any overlapping outputs into the current output buffer
        r = self.y_buffer
        if r is not None:
            overlap = max(min(r.shape[-1] - self.y_offset, self.overlap), 0)
            r[..., self.y_offset : self.y_offset + overlap] += y[..., :overlap]
            y = pt.cat([r, y[..., overlap:]], dim=-1)
        self.y_offset += self.stride * x.shape[-1]

        if final:
            # if no more inputs are available, flush the output buffer
            y = y[..., : max(y.shape[-1] - self.padding, 0)]
            self.y_buffer = None
            self.y_offset = 0
        else:
            # otherwise, save off the current outputs for further accumulation
            offset = min(self.y_offset, y.shape[-1] - self.padding)
            y, self.y_buffer = y[..., :offset], y[..., offset:]
            self.y_offset -= offset

        # add in the bias term after all accumulation
        bias: pt.Tensor | None = self.bias
        if bias is not None:
            with pt.no_grad():
                y = y + bias

        return y if y.shape[-1] != 0 else None


class Pool1dStream(BaseStream):
    """Streamer for PyTorch pooling modules (AvgPool1d/MaxPool1d)

    Args:
        net: pooling layer to stream

    Raises:
        ValueError: an invalid pooling layer was passed

    """

    x_buffer: list[pt.Tensor]

    def __init__(self, net: nn.AvgPool1d | nn.MaxPool1d):
        super().__init__()

        if isinstance(net, nn.AvgPool1d):
            self.pool_type = "Avg"
        elif isinstance(net, nn.MaxPool1d):
            self.pool_type = "Max"
        else:
            raise ValueError("invalid_pool_module")

        k, s = net.kernel_size, net.stride
        self.field = k[0] if isinstance(k, tuple) else k
        self.stride = s[0] if isinstance(s, tuple) else s

        self.x_buffer = []
        self.x_length = 0
        self.x_skip = 0

    def __repr__(self) -> str:
        t, f, s = self.pool_type, self.field, self.stride
        return f"{t}Pool1dStream(kernel_size={f}, stride={s})"

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        y: pt.Tensor | None = None

        # buffer inputs until we fill the receptive field
        self.x_buffer.append(x)
        self.x_length += x.shape[-1]
        if self.x_length - self.x_skip >= self.field:
            # stack the buffered inputs for pooling
            x = pt.cat(self.x_buffer, dim=-1)[..., self.x_skip :]
            self.x_buffer.clear()
            self.x_length = 0

            # calculate sizes of the inputs/outputs
            steps = (x.shape[-1] - self.field) // self.stride
            valid = self.field + steps * self.stride
            next_ = (steps + 1) * self.stride

            # apply the pooling operation
            if self.pool_type == "Avg":
                y = ptf.avg_pool1d(
                    x[..., :valid].unsqueeze(0),
                    kernel_size=self.field,
                    stride=self.stride,
                ).squeeze(0)
            elif self.pool_type == "Max":
                y = ptf.max_pool1d(
                    x[..., :valid].unsqueeze(0),
                    kernel_size=self.field,
                    stride=self.stride,
                ).squeeze(0)

            # save off any inputs needed for the next application
            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)
            self.x_skip = max(next_ - x.shape[-1], 0)

        return y


class Elementwise1dStream(BaseStream):
    """Streamer for simple pointwise/elementwise PyTorch modules

    Elementwise modules are modules that do not have a spatial receptive
    field, so operations do not span or stride the final dimension.

    Args:
        net: PyTorch layer to stream

    """

    def __init__(self, net: nn.Module):
        super().__init__()

        self.net = net

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> pt.Tensor | None:
        with pt.no_grad():
            y = self.net(x.unsqueeze(0)).squeeze(0)

        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError("invalid_elementwise_output")

        return y


def _conv1d_stream(net: nn.Conv1d) -> BaseStream:
    if net.kernel_size[0] == 1 and net.stride[0] == 1:
        return Elementwise1dStream(net)
    else:
        return Conv1dStream(net)


def _convtranspose1d_stream(net: nn.ConvTranspose1d) -> BaseStream:
    if net.kernel_size[0] == 1 and net.stride[0] == 1:
        return Elementwise1dStream(net)
    else:
        return ConvTranspose1dStream(net)


def register_streamer(
    module: type,
    streamer: T.Callable[[nn.Module], BaseStream],
) -> None:
    """Registers a new stream for a PyTorch module type

    Note that custom streamers are not needed for simple elementwise modules,
    these are wrapped automatically with Elementwise1dStream instances. Custom
    streamers are only needed for custom modules that process more than one
    spatial element at a time.

    Args:
        module: the type of the PyTorch module to register for
        streamer: a callable that constructs a streamer for this module type

    Example:
        ```python
        class MyModule(nn.Module):
            ...

        class MyModuleStream(pts.BaseStream):
            def __init__(self, net: MyModule):
            ...

        register_streamer(MyModule, lambda net: MyModuleStream(net))
        ```

    """
    for i, (m, _) in enumerate(_stream_map):
        if m == module:
            _stream_map[i] = (module, streamer)
            return
    _stream_map.append((module, streamer))


def create_stream(net: nn.Module) -> BaseStream:
    """Finds and creates a stream handler for a registered or built-in module

    Args:
        net: the module to stream

    Returns:
        stream: a streamer that wraps the specified module

    Example:
        ```python
        stream = create_stream(nn.Conv1d(1, 128, 1))
        ```

    """

    for module, streamer in _stream_map:
        if isinstance(net, module):
            return streamer(net)
    return Elementwise1dStream(net)


_stream_map: list[tuple[type, T.Callable[[nn.Module], BaseStream]]] = [
    (
        nn.Sequential,
        lambda net: Sequential1dStream(T.cast(nn.Sequential, net)),
    ),
    (
        Residual1d,
        lambda net: Residual1dStream(T.cast(Residual1d, net)),
    ),
    (
        nn.Conv1d,
        lambda net: _conv1d_stream(T.cast(nn.Conv1d, net)),
    ),
    (
        nn.ConvTranspose1d,
        lambda net: _convtranspose1d_stream(T.cast(nn.ConvTranspose1d, net)),
    ),
    (
        nn.AvgPool1d,
        lambda net: Pool1dStream(T.cast(nn.AvgPool1d, net)),
    ),
    (
        nn.MaxPool1d,
        lambda net: Pool1dStream(T.cast(nn.MaxPool1d, net)),
    ),
]
