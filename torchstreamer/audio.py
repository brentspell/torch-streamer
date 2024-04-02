import math
import typing as T

import torch as pt
import torch.nn.functional as ptf
import torchaudio as pta

from . import core as pts


class ResampleStream(pts.BaseStream):
    x_buffer: T.List[pt.Tensor]

    def __init__(
        self,
        source_fs: int,
        target_fs: int,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float = 14.769656459379492,
    ):
        super().__init__()

        gcd = math.gcd(source_fs, target_fs)
        self.source_rate = source_fs // gcd
        self.target_rate = target_fs // gcd

        kernel, width = pta.functional.functional._get_sinc_resample_kernel(
            orig_freq=source_fs,
            new_freq=target_fs,
            gcd=gcd,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            resampling_method=resampling_method,
            beta=beta,
            device=pt.zeros([]).device,
        )
        self.register_buffer("kernel", kernel)

        lpad = pt.zeros([1, width])
        rpad = pt.zeros([1, width + self.source_rate])
        self.register_buffer("lpad", lpad)
        self.register_buffer("rpad", rpad)

        self.channels = 0
        self.field = kernel.shape[2]
        self.stride = self.source_rate

        self.started = False
        self.x_buffer = []
        self.x_length = 0
        self.x_total = 0
        self.y_total = 0

    def forward(self, x: pt.Tensor, final: bool = False) -> T.Optional[pt.Tensor]:
        if len(x.shape) == 1:
            self.channels = 0
            x = x.unsqueeze(0)
        else:
            self.channels = x.shape[0]

        if not self.started:
            self.x_buffer.append(self.lpad.repeat(max(self.channels, 1), 1))
            self.x_length += self.lpad.shape[-1]
            self.started = True

        self.x_buffer.append(x)
        self.x_length += x.shape[-1]
        self.x_total += x.shape[-1]

        if final:
            self.x_buffer.append(self.rpad.repeat(max(self.channels, 1), 1))
            self.x_length += self.rpad.shape[-1]

        y: T.Optional[pt.Tensor] = None
        if self.x_length >= self.field:
            x = pt.cat(self.x_buffer, dim=-1)
            self.x_buffer.clear()
            self.x_length = 0

            steps = (x.shape[-1] - self.field) // self.stride
            valid = self.field + steps * self.stride
            next_ = (steps + 1) * self.stride

            y = ptf.conv1d(
                x[..., :valid].unsqueeze(1),
                weight=self.kernel,
                stride=self.stride,
            )

            self.x_buffer.append(x[..., next_:])
            self.x_length += max(x.shape[-1] - next_, 0)

            y = y.mT.reshape(y.shape[0], -1)
            self.y_total += y.shape[-1]

            if final:
                y_total = math.ceil(self.target_rate * self.x_total / self.source_rate)
                if self.y_total > y_total:
                    y = y[..., : -(self.y_total - y_total)]

            if self.channels == 0:
                y = y.squeeze(0)

        return y
