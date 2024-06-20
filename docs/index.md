# Overview

The Torch-Streamer library provides support for performing exact inference
of PyTorch neural neural networks over partial/streamed 1D inputs. It provides
stream handlers for a variety of built-in PyTorch modules
(convolution/transposed convolution, pooling, sequential networks, etc), each
of which performs the necessary internal buffering required to process
streaming inputs.

Streamers can run on the GPU just like ordinary PyTorch modules, and they also
support scripting for efficient fused operations.

## Installation

Torch-Streamer can be installed from
[PyPI](https://pypi.org/project/torch-streamer/) using
[pip](https://github.com/pypa/pip) or [uv](https://github.com/astral-sh/uv).

```
pip install torch-streamer
```

## Basic Usage

The following example shows how to stream a simple 1D convolution. It streams
blocks of `BLOCK_SIZE` elements through the convolutional filter with no
padding. Batched inputs are not supported, so note that input tensors are
shaped `[C, T]`,

```python
import torch as pt
import torchstreamer as pts

BLOCK_SIZE = 100

# create the convolution module
conv = pt.nn.Conv1d(64, 128, 3, stride=2)

# create a stream for the convolution
stream = pts.Conv1dStream(conv)

# create a random 1D input tensor
x = pt.randn([64, 1000])

# apply the convolution directly to the whole input tensor
yc = conv(x)

# stream the input tensor through the convolution, one block at a time
ys = pt.cat(
    [
        stream(x[..., i : i + BLOCK_SIZE])
        for i in range(0, x.shape[-1], BLOCK_SIZE)
    ],
    dim=-1
)

# verify the outputs match
assert pt.allclose(yc, ys, atol=1e-6)
```

Note that all stream handlers are themselves PyTorch modules, so they can
be treated as callables. The callable accepts an input tensor and optionally
returns an output tensor if enough data has been buffered to produce an
output (`None` otherwise).

## Sequential Networks

Complex networks built using `torch.nn.Sequential` can used to compose
streams that work over the whole network, assuming each layer in the network
is supported or has a registered streamer. See [below](#custom-stream-handlers)
for more information on stream customization.

```python
import torch as pt
import torchstreamer as pts
from torch import nn

net = nn.Sequential(
    nn.Conv1d(1, 128, 1),
    nn.GELU(),
    nn.Conv1d(128, 128, 2, stride=2),
    nn.GELU(),
    nn.Conv1d(128, 128, 2, stride=2),
    nn.AvgPool1d(2),
    nn.ConvTranspose1d(128, 128, 2, stride=2),
    nn.GELU(),
    nn.ConvTranspose1d(128, 128, 2, stride=2),
    nn.GELU(),
    nn.Conv1d(128, 1, 1),
)

stream = pts.Sequential1dStream(net)
```

## Residual Networks
ResNets can require special handling, since the residual added to the output
may be a different length due to the network's receptive field, since edge
padding is not applied. `torchstreamer` provides a simple
[Residual1d](reference.md#torchstreamer.Residual1d) module that can be
used to build residual networks that work automatically with streaming. Of
course you can also build your own streamer for your residual modules if
you don't have control over the model's architecture.

```python
import torch as pt
import torch.nn as nn
import torchstreamer as pts

net = nn.Sequential(
    nn.Conv1d(1, 128, 1),
    *[
        pts.Residual1d(
            nn.Sequential(
                nn.GELU(),
                nn.Conv1d(128, 256, 3),
                nn.GELU(),
                nn.Conv1d(256, 128, 1),
            )
        )
        for _ in range(3)
    ],
    nn.Conv1d(128, 1, 1),
)

stream = pts.Sequential1dStream(net)
```

## Custom Stream Handlers
In order to work with the built-in `Sequential1d` streamer, custom PyTorch
modules may need to register a corresponding stream handler. This is not
necessary for simple element-wise modules, since they have a receptive
field of 1 and produce a single output element for each input element.

A custom stream handler can be registered with the
[register_streamer](reference.md#torchstreamer.register_streamer)
function. This function accepts the type of the PyTorch module to handle
and a callable that can construct the streamer for the module. The following
example demonstrates how to do this for a custom ResNet block.

```python
import typing as T

import torch as pt
import torchstreamer as pts

# create a simple residual module
class MyResBlock(pt.nn.Module):
    def __init__(self, inner: pt.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        return x + self.inner(x)

# create a custom stream handler for the residual module
class MyResBlockStream(pts.BaseStream):
    def __init__(self, net: MyResBlock):
        super().__init__()
        self.net = net

    def forward(self, x: pt.Tensor, final: bool = False) -> T.Optional[pt.Tensor]:
        # apply the inner network of the residual module
        y = self.net.inner(x.unsqueeze(0))

        # accommodate the network's receptive field by stripping
        # elements from the edge of the residual connection
        pad = (x.shape[-1] - y.shape[-1]) // 2
        res = x[..., pad:-pad]

        # add the residual connection to the outputs
        return y.squeeze(0) + res

# register the streamer with the library
pts.register_streamer(MyResBlock, lambda net: MyResBlockStream(net))

# create the neural network containing the custom resblock module
net = pt.nn.Sequential(
    pt.nn.Conv1d(1, 32, 1),
    MyResBlock(pt.nn.Conv1d(32, 32, 3)),
    pt.nn.Conv1d(32, 1, 1),
)

# create a stream that uses the custom stream handler
# whenever it encounters an instance of MyResBlock
stream = pts.Sequential1dStream(net)
```
