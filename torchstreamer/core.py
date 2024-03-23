import abc
import typing as T

import torch as pt
import torch.nn.functional as ptf


class BaseStream(pt.nn.Module, abc.ABC):
    @abc.abstractmethod
    def write(self, x: pt.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError()

    def generate(self, xs: T.Iterable[pt.Tensor]) -> T.Iterable[pt.Tensor]:
        for x in xs:
            self.write(x)
            y = self.read()
            if y is not None:
                yield y

        y = self.read(final=True)
        if y is not None:
            yield y


class Sequential1dStream(BaseStream):
    def __init__(self, net: pt.nn.Sequential):
        super().__init__()

        self.streams = pt.nn.ModuleList([module1d_stream(m) for m in net.children()])

    @pt.jit.export
    def write(self, x: pt.Tensor) -> None:
        x_: T.Optional[pt.Tensor] = x
        for i, s in enumerate(self.streams):
            if x_ is not None:
                s.write(x_)
            if i != len(self.streams) - 1:
                x_ = s.read()
                if x_ is None:
                    return

    @pt.jit.export
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        x: T.Optional[pt.Tensor] = None
        y: T.Optional[pt.Tensor] = None
        for s in self.streams:
            if x is not None:
                s.write(x)
            y = s.read(final=final)
            x = y
        return y

    @pt.jit.export
    def clear(self) -> None:
        for s in self.streams:
            s.clear()


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

    @pt.jit.export
    def write(self, x: pt.Tensor) -> None:
        self.x_buffer.append(x)
        self.stream.write(x)

    @pt.jit.export
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        y = self.stream.read(final=final)
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

    @pt.jit.export
    def clear(self) -> None:
        self.stream.clear()
        self.x_buffer.clear()
        self.x_offset = 0


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
    y_buffer: T.List[pt.Tensor]

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
        self.y_buffer = []
        self.x_length = 0
        self.x_skip = 0

    def __repr__(self) -> str:
        o, i, k = self.weight.shape
        return f"Conv1dStream({i}, {o}, kernel_size={k}, stride={self.stride})"

    @pt.jit.export
    def write(self, x: pt.Tensor) -> None:
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
            self.y_buffer.append(y)

            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)
            self.x_skip = max(next_ - x.shape[-1], 0)

    @pt.jit.export
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = None
        if self.y_buffer:
            y = pt.cat(self.y_buffer, dim=-1)
            self.y_buffer.clear()
        return y

    @pt.jit.export
    def clear(self) -> None:
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.x_length = 0
        self.x_skip = 0


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
        self.w_offset = 0

    def __repr__(self) -> str:
        i, o, k = self.weight.shape
        return f"ConvTranspose1dStream({i}, {o}, kernel_size={k}, stride={self.stride})"

    @pt.jit.export
    def write(self, x: pt.Tensor) -> None:
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
        if r is None:
            self.y_buffer = y
            self.w_offset = 0
        else:
            overlap = max(min(r.shape[-1] - self.w_offset, self.overlap), 0)
            r[..., self.w_offset : self.w_offset + overlap] += y[..., :overlap]
            self.y_buffer = pt.cat([r, y[..., overlap:]], dim=-1)
        self.w_offset += self.stride * x.shape[-1]

    @pt.jit.export
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = self.y_buffer
        if y is not None:
            if final:
                y = y[..., : max(y.shape[-1] - self.padding, 0)]
                self.y_buffer = None
                self.w_offset = 0
            else:
                offset = min(self.w_offset, y.shape[-1] - self.padding)
                y, self.y_buffer = y[..., :offset], y[..., offset:]
                self.w_offset -= offset

        if y is not None and y.shape[-1] == 0:
            y = None

        bias: T.Optional[pt.Tensor] = self.bias
        if y is not None and bias is not None:
            with pt.no_grad():
                y = y + bias

        return y

    @pt.jit.export
    def clear(self) -> None:
        self.y_buffer = None
        self.w_offset = 0


class Elementwise1dStream(BaseStream):
    y_buffer: T.List[pt.Tensor]

    def __init__(self, net: pt.nn.Module):
        super().__init__()

        self.net = net
        self.y_buffer = []

    @pt.jit.export
    def write(self, x: pt.Tensor) -> None:
        with pt.no_grad():
            y = self.net(x.unsqueeze(0)).squeeze(0)

        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError("invalid_elementwise_output")

        self.y_buffer.append(y)

    @pt.jit.export
    def read(self, final: bool = False) -> T.Optional[pt.Tensor]:
        y: T.Optional[pt.Tensor] = None
        if self.y_buffer:
            y = pt.cat(self.y_buffer, dim=-1)
            self.y_buffer.clear()
        return y

    @pt.jit.export
    def clear(self) -> None:
        self.y_buffer.clear()


def module1d_stream(net: pt.nn.Module) -> BaseStream:
    for module, streamer in _stream_map:
        if isinstance(net, module):
            return streamer(net)
    return Elementwise1dStream(net)


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
        lambda net: Conv1dStream(T.cast(pt.nn.Conv1d, net)),
    ),
    (
        pt.nn.ConvTranspose1d,
        lambda net: ConvTranspose1dStream(
            T.cast(pt.nn.ConvTranspose1d, net),
        ),
    ),
]
