import hypothesis
import hypothesis.strategies as hypstrat
import torch as pt
import torchaudio as pta

import torchstreamer as pts
import torchstreamer.audio as ptsa


@hypothesis.given(
    source_fs=hypstrat.sampled_from([8000, 16000, 22050, 44100, 48000]),
    target_fs=hypstrat.sampled_from([8000, 16000, 22050, 44100, 48000]),
    channels=hypstrat.integers(min_value=0, max_value=2),
    resampling_method=hypstrat.sampled_from(["sinc_interp_hann", "sinc_interp_kaiser"]),
    lowpass_filter_width=hypstrat.integers(min_value=1, max_value=20),
    rolloff=hypstrat.floats(min_value=0.01, max_value=0.99),
    beta=hypstrat.floats(min_value=1.0, max_value=50.0),
    length=hypstrat.integers(min_value=1, max_value=100),
    block=hypstrat.integers(min_value=1, max_value=10),
    script=hypstrat.booleans(),
)
@hypothesis.example(
    source_fs=16000,
    target_fs=48000,
    channels=1,
    resampling_method="sinc_interp_hann",
    lowpass_filter_width=6,
    rolloff=0.99,
    beta=14.769656459379492,
    length=100,
    block=1,
    script=False,
)
def test_resample(
    source_fs: int,
    target_fs: int,
    channels: int,
    resampling_method: str,
    lowpass_filter_width: int,
    rolloff: float,
    beta: float,
    length: int,
    block: int,
    script: bool,
) -> None:
    hypothesis.assume(source_fs != target_fs)

    x = pt.randn([channels, length] if channels > 0 else [length]).clamp(-1.0, 1.0)
    s = ptsa.ResampleStream(
        source_fs=source_fs,
        target_fs=target_fs,
        resampling_method=resampling_method,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        beta=beta,
    )
    if script:
        s = pt.jit.script(s)

    yc = pta.functional.resample(
        x,
        source_fs,
        target_fs,
        resampling_method=resampling_method,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        beta=beta,
    )

    ys = do_stream(s, x, block)
    assert ys.shape == yc.shape
    assert pt.allclose(yc, ys, atol=1e-4)


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
