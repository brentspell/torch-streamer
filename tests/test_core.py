import typing as T

import hypothesis
import hypothesis.strategies as hypstrat
import pytest
import torch as pt
import torch.nn as nn

import torchstreamer as pts

CH = 2  # avoid any accidental broadcasting with channels=1


@hypothesis.given(
    length=hypstrat.integers(min_value=10, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_sequential1d(
    length: int,
    block: int,
    script: bool,
) -> None:
    x = pt.randn([CH, length]).clamp(-1, 1)
    m = nn.Sequential(
        nn.ConvTranspose1d(CH, CH, kernel_size=3, stride=2),
        nn.ConvTranspose1d(CH, CH, kernel_size=1),
        nn.Conv1d(CH, CH, kernel_size=1),
        nn.Conv1d(CH, 2 * CH, kernel_size=3, dilation=2),
        nn.Conv1d(2 * CH, CH, kernel_size=2),
        nn.Conv1d(CH, CH, kernel_size=1, stride=2),
        nn.AvgPool1d(kernel_size=3, stride=2),
        nn.MaxPool1d(kernel_size=3, stride=2),
        nn.ReLU(),
    )

    s = pts.Sequential1dStream(m)
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


@hypothesis.given(
    length=hypstrat.integers(min_value=10, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_residual1d(
    length: int,
    block: int,
    script: bool,
) -> None:
    x = pt.randn([CH, length]).clamp(-1, 1)
    m = pts.Residual1d(
        nn.Sequential(
            nn.Conv1d(CH, CH, kernel_size=1),
            nn.Conv1d(CH, CH, kernel_size=3, dilation=2),
            nn.Conv1d(CH, CH, kernel_size=2),
        )
    )

    s = pts.Residual1dStream(m)
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


def test_residual1d_invalid() -> None:
    with pytest.raises(ValueError, match=r"invalid_residual_stride"):
        pts.Residual1d(nn.Conv1d(CH, CH, kernel_size=1, stride=2))


@hypothesis.given(
    kernel=hypstrat.integers(min_value=1, max_value=10),
    stride=hypstrat.integers(min_value=1, max_value=10),
    dilation=hypstrat.integers(min_value=1, max_value=10),
    bias=hypstrat.booleans(),
    length=hypstrat.integers(min_value=0, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_conv1d(
    kernel: int,
    stride: int,
    dilation: int,
    bias: bool,
    length: int,
    block: int,
    script: bool,
) -> None:
    length = max(length, dilation * (kernel - 1) + 1)

    x = pt.randn([CH, length]).clamp(-1, 1)
    m = nn.Conv1d(
        CH,
        CH,
        kernel_size=kernel,
        stride=stride,
        dilation=dilation,
        groups=CH,
        bias=bias,
    )
    s = pts.Conv1dStream(m)
    assert repr(s).startswith("Conv1dStream")
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


@hypothesis.given(
    kernel=hypstrat.integers(min_value=1, max_value=10),
    stride=hypstrat.integers(min_value=1, max_value=10),
    dilation=hypstrat.integers(min_value=1, max_value=10),
    bias=hypstrat.booleans(),
    length=hypstrat.integers(min_value=1, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_conv_transpose1d(
    kernel: int,
    stride: int,
    dilation: int,
    bias: bool,
    length: int,
    block: int,
    script: bool,
) -> None:
    x = pt.randn([CH, length]).clamp(-1, 1)
    m = nn.ConvTranspose1d(
        CH,
        CH,
        kernel_size=kernel,
        stride=stride,
        dilation=dilation,
        groups=CH,
        bias=bias,
    )
    s = pts.ConvTranspose1dStream(m)
    assert repr(s).startswith("ConvTranspose1dStream")
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


@hypothesis.given(
    pool_type=hypstrat.sampled_from([nn.AvgPool1d, nn.MaxPool1d]),
    kernel=hypstrat.integers(min_value=1, max_value=10),
    stride=hypstrat.integers(min_value=1, max_value=10),
    length=hypstrat.integers(min_value=0, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_pool1d(
    pool_type: type,
    kernel: int,
    stride: int,
    length: int,
    block: int,
    script: bool,
) -> None:
    length = max(length, kernel)

    x = pt.randn([CH, length]).clamp(-1, 1)
    m = pool_type(
        kernel_size=kernel,
        stride=stride,
    )
    s = pts.Pool1dStream(m)
    assert repr(s).startswith(f"{pool_type.__name__}Stream")
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


def test_pool1d_invalid() -> None:
    with pytest.raises(ValueError, match=r"invalid_pool_module"):
        m = nn.Conv1d(CH, CH, kernel_size=1)
        pts.Pool1dStream(T.cast(nn.AvgPool1d, m))


@hypothesis.given(
    length=hypstrat.integers(min_value=1, max_value=20),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
def test_elementwise1d(
    length: int,
    block: int,
    script: bool,
) -> None:
    x = pt.randn([CH, length]).clamp(-1, 1)
    m = nn.Conv1d(CH, CH, kernel_size=1)
    s = pts.Elementwise1dStream(m)
    if script:
        s = pt.jit.script(s)

    yc = m(x)
    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


def test_elementwise1d_invalid() -> None:
    m = nn.Conv1d(CH, CH, kernel_size=2)
    s = pts.Elementwise1dStream(m)

    with pytest.raises(RuntimeError, match=r"invalid_elementwise_output"):
        s(pt.zeros([CH, 2]))


def test_custom() -> None:
    class MyResBlock(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x: pt.Tensor) -> pt.Tensor:
            return x + self.inner(x)

    pts.register_streamer(MyResBlock, lambda net: pts.Residual1dStream(net.inner))
    pts.register_streamer(MyResBlock, lambda net: pts.Residual1dStream(net.inner))

    x = pt.randn([CH, 10])
    m = nn.Sequential(
        nn.Conv1d(CH, CH, kernel_size=1),
        MyResBlock(
            nn.Sequential(
                nn.Conv1d(CH, CH, kernel_size=3, padding=1),
                nn.Conv1d(CH, CH, kernel_size=5, padding=2),
            )
        ),
        nn.Conv1d(CH, CH, kernel_size=1),
    )
    s = pts.Sequential1dStream(m)

    yc = m(x)[..., 3:-3]
    ys = do_stream(s, x, b=1)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-6)


def do_stream(
    stream: pts.BaseStream | pt.jit.ScriptModule,
    x: pt.Tensor,
    b: int,
) -> pt.Tensor:
    ys = []
    for i in range(0, x.shape[-1], b):
        y = stream(x[..., i : i + b], final=i + b >= x.shape[-1])
        if y is not None:
            assert y.shape[-1] != 0
            ys.append(y)

    return pt.cat(list(ys), dim=-1)
