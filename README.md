# [funnel](https://tschm.github.io/funnel)

[![PyPI version](https://badge.fury.io/py/funnel.svg)](https://badge.fury.io/py/funnel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![CI](https://github.com/tschm/funnel/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/funnel/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/funnel/badge.svg?branch=main)](https://coveralls.io/github/tschm/funnel?branch=main)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/package)

## Getting Started

For later...

### **Set Up Environment**

```bash
task build:install
```

This installs/updates [uv](https://github.com/astral-sh/uv),
creates your virtual environment and installs dependencies.

For adding or removing packages:

```bash
uv add/remove requests  # for main dependencies
uv add/remove requests --dev  # for dev dependencies
```

### **Configure Pre-commit Hooks**

```bash
task quality:fmt
```

Installs hooks to maintain code quality and formatting.

### **Update Project Info**

- Edit `pyproject.toml` to update authors and email addresses

## Development Commands

```bash
task docs:tests   # Run test suite
task docs:marimo  # Start Marimo notebooks
```

## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
