import abc
import typing as T

import torch as pt
import torch.nn.functional as ptf


class BaseStream(pt.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> T.Optional[pt.Tensor]:
        raise NotImplementedError()


class Sequential1dStream(BaseStream):
    def __init__(self, net: pt.nn.Sequential):
        super().__init__()

        self.streams = pt.nn.ModuleList([module1d_stream(m) for m in net.children()])

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> T.Optional[pt.Tensor]:
        x_: T.Optional[pt.Tensor] = x
        for i, stream in enumerate(self.streams):
            if x_ is not None:
                x_ = stream(x_, final=final)
        return x_


class Residual1dStream(BaseStream):
    x_buffer: T.List[pt.Tensor]

    def __init__(self, net: pt.nn.Module):
        super().__init__()

        convs = [m for m in net.modules() if isinstance(m, pt.nn.Conv1d)]
        field = (
            sum(
                convs[i].dilation[0] * (convs[i].kernel_size[0] - 1)
                for i in range(len(convs))
            )
            + 1
        )
        self.padding = (field - 1) // 2

        self.stream = module1d_stream(net.net if isinstance(net, Residual1d) else net)

        self.x_buffer = []
        self.x_offset = 0

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> T.Optional[pt.Tensor]:
        self.x_buffer.append(x)
        y = self.stream(x, final=final)

        if y is not None:
            x = pt.cat(self.x_buffer, dim=-1)
            self.x_buffer.clear()

            offset = max(min(self.padding - self.x_offset, x.shape[-1]), 0)
            if offset > 0:
                x = x[..., offset:]
                self.x_offset = offset

            y += x[..., : y.shape[-1]]
            self.x_buffer.append(x[..., y.shape[-1] :])

        return y


class Residual1d(pt.nn.Module):
    def __init__(self, net: pt.nn.Module):
        super().__init__()

        if any(m.stride[0] != 1 for m in net.modules() if isinstance(m, pt.nn.Conv1d)):
            raise ValueError("invalid_residual_stride")

        self.net = net

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        y = self.net(x)
        p = x.shape[-1] - y.shape[-1]
        if p > 0:
            x = x[..., p // 2 : -(p // 2 + p % 2)]
        return y + x


class Conv1dStream(BaseStream):
    weight: pt.nn.Parameter
    bias: T.Optional[pt.nn.Parameter]
    x_buffer: T.List[pt.Tensor]

    def __init__(self, net: pt.nn.Conv1d):
        super().__init__()

        self.register_parameter(
            "weight",
            pt.nn.Parameter(
                pt._weight_norm(net.weight_v, net.weight_g)
                if hasattr(net, "weight_g")
                else net.weight
            ),
        )
        self.register_parameter(
            "bias",
            pt.nn.Parameter(net.bias) if net.bias is not None else None,
        )

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
    ) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = None
        self.x_buffer.append(x)
        self.x_length += x.shape[-1]
        if self.x_length - self.x_skip >= self.field:
            x = pt.cat(self.x_buffer, dim=-1)[..., self.x_skip :]
            self.x_buffer.clear()
            self.x_length = 0

            steps = (x.shape[-1] - self.field) // self.stride
            valid = self.field + steps * self.stride
            next_ = (steps + 1) * self.stride

            with pt.no_grad():
                y = ptf.conv1d(
                    x[..., :valid].unsqueeze(0),
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                ).squeeze(0)

            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)
            self.x_skip = max(next_ - x.shape[-1], 0)

        return y


class ConvTranspose1dStream(BaseStream):
    weight: pt.nn.Parameter
    bias: T.Optional[pt.nn.Parameter]
    y_buffer: T.Optional[pt.Tensor]

    def __init__(self, net: pt.nn.ConvTranspose1d):
        super().__init__()

        self.register_parameter(
            "weight",
            pt.nn.Parameter(
                pt._weight_norm(net.weight_v, net.weight_g)
                if hasattr(net, "weight_g")
                else net.weight
            ),
        )
        self.register_parameter(
            "bias",
            pt.nn.Parameter(net.bias.unsqueeze(-1)) if net.bias is not None else None,
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
    ) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = None
        with pt.no_grad():
            y = ptf.conv_transpose1d(
                x.unsqueeze(0),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            ).squeeze(0)

        y = ptf.pad(y, [0, self.padding])

        r = self.y_buffer
        if r is not None:
            overlap = max(min(r.shape[-1] - self.y_offset, self.overlap), 0)
            r[..., self.y_offset : self.y_offset + overlap] += y[..., :overlap]
            y = pt.cat([r, y[..., overlap:]], dim=-1)
        self.y_offset += self.stride * x.shape[-1]

        if final:
            y = y[..., : max(y.shape[-1] - self.padding, 0)]
            self.y_buffer = None
            self.y_offset = 0
        else:
            offset = min(self.y_offset, y.shape[-1] - self.padding)
            y, self.y_buffer = y[..., :offset], y[..., offset:]
            self.y_offset -= offset

        bias: T.Optional[pt.Tensor] = self.bias
        if bias is not None:
            with pt.no_grad():
                y = y + bias

        return y if y.shape[-1] != 0 else None


class Pool1dStream(BaseStream):
    x_buffer: T.List[pt.Tensor]

    def __init__(self, net: T.Union[pt.nn.AvgPool1d, pt.nn.MaxPool1d]):
        super().__init__()

        if isinstance(net, pt.nn.AvgPool1d):
            self.pool_type = "Avg"
        elif isinstance(net, pt.nn.MaxPool1d):
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
    ) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = None
        self.x_buffer.append(x)
        self.x_length += x.shape[-1]
        if self.x_length - self.x_skip >= self.field:
            x = pt.cat(self.x_buffer, dim=-1)[..., self.x_skip :]
            self.x_buffer.clear()
            self.x_length = 0

            steps = (x.shape[-1] - self.field) // self.stride
            valid = self.field + steps * self.stride
            next_ = (steps + 1) * self.stride

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

            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)
            self.x_skip = max(next_ - x.shape[-1], 0)

        return y


class Elementwise1dStream(BaseStream):
    def __init__(self, net: pt.nn.Module):
        super().__init__()

        self.net = net

    def forward(
        self,
        x: pt.Tensor,
        final: bool = False,
    ) -> T.Optional[pt.Tensor]:
        with pt.no_grad():
            y = self.net(x.unsqueeze(0)).squeeze(0)

        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError("invalid_elementwise_output")

        return y


def module1d_stream(net: pt.nn.Module) -> BaseStream:
    for module, streamer in _stream_map:
        if isinstance(net, module):
            return streamer(net)
    return Elementwise1dStream(net)


def conv1d_stream(net: pt.nn.Conv1d) -> BaseStream:
    if net.kernel_size[0] == 1 and net.stride[0] == 1:
        return Elementwise1dStream(net)
    else:
        return Conv1dStream(net)


def convtranspose1d_stream(net: pt.nn.ConvTranspose1d) -> BaseStream:
    if net.kernel_size[0] == 1 and net.stride[0] == 1:
        return Elementwise1dStream(net)
    else:
        return ConvTranspose1dStream(net)


def register_streamer(
    module: type,
    streamer: T.Callable[[pt.nn.Module], BaseStream],
) -> None:
    for i, (m, _) in enumerate(_stream_map):
        if m == module:
            _stream_map[i] = (module, streamer)
            return
    _stream_map.append((module, streamer))


_stream_map: T.List[T.Tuple[type, T.Callable[[pt.nn.Module], BaseStream]]] = [
    (
        pt.nn.Sequential,
        lambda net: Sequential1dStream(T.cast(pt.nn.Sequential, net)),
    ),
    (
        Residual1d,
        lambda net: Residual1dStream(T.cast(Residual1d, net)),
    ),
    (
        pt.nn.Conv1d,
        lambda net: conv1d_stream(T.cast(pt.nn.Conv1d, net)),
    ),
    (
        pt.nn.ConvTranspose1d,
        lambda net: convtranspose1d_stream(T.cast(pt.nn.ConvTranspose1d, net)),
    ),
    (
        pt.nn.AvgPool1d,
        lambda net: Pool1dStream(T.cast(pt.nn.AvgPool1d, net)),
    ),
    (
        pt.nn.MaxPool1d,
        lambda net: Pool1dStream(T.cast(pt.nn.MaxPool1d, net)),
    ),
]
