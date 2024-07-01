# Torch-Streamer
This package implements a framework for streaming 1D convolutions in PyTorch
without padding or pseudo-streaming/cross-fading.

## Status
[![PyPI](https://img.shields.io/pypi/v/torch-streamer.svg)](https://pypi.org/project/torch-streamer)
[![Tests](https://github.com/brentspell/torch-streamer/actions/workflows/test.yml/badge.svg)](https://github.com/brentspell/torch-streamer/actions/workflows/test.yml)
[![Coveralls](https://coveralls.io/repos/github/brentspell/torch-streamer/badge.svg?branch=main)](https://coveralls.io/github/brentspell/torch-streamer?branch=main)
[![Docs](https://readthedocs.org/projects/torch-streamer/badge/?version=latest)](https://torch-streamer.readthedocs.io/en/latest/?badge=latest)

## Usage

### Install with pip
```bash
pip install torch-streamer
```

### Documentation

Docs are available at
[torch-streamer.readthedocs.io](https://torch-streamer.readthedocs.io/en/latest).

## Development

### Setup
The following script creates a virtual environment using
[pyenv](https://github.com/pyenv/pyenv) for the project and installs
dependencies with [uv](https://pypi.org/project/uv/).

```bash
pyenv install 3.10
pyenv virtualenv 3.10 torch-streamer
bin/deps
```

You can also use [pre-commit](https://pre-commit.com/) with the project to
run tests, etc. at commit time.

```bash
pre-commit install
```

### Testing
Testing, formatting, and static checking can all be done with pre-commit at
any time.

```bash
pre-commit run --all-files
```

There is also a watcher script that can be used to run these whenever a file
changes.

```bash
bin/watch
```

### Documentation
The project uses [MkDocs](https://www.mkdocs.org/) with
[mkdocstrings](https://mkdocstrings.github.io/python/) for documentation, and
you can start a mkdocs web server to test/edit documentation as follows.

```bash
bin/docserve
```

Documentation is hosted by
[Read the Docs](https://torch-streamer.readthedocs.io/en/latest) and will
automatically update when the main branch is merged.

### Releasing
The library can be updated on the main PyPI repository as follows.

```bash
bin/release pypi
```

If needed, you can release to the test PyPI repository with this command.

```
bin/release pypi-test
```

## License
Copyright © 2024 Brent M. Spell

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
