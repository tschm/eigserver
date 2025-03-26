# eigserver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MARIMO](https://github.com/tschm/eigserver/actions/workflows/marimo.yml/badge.svg)](https://github.com/tschm/eigserver/actions/workflows/marimo.yml)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/experiments)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/eigserver)

## Getting Started

### **Set Up Environment**

```bash
make install
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
make fmt
```

Installs hooks to maintain code quality and formatting.

## Development Commands

```bash
make marimo  # Start Marimo notebooks
```

## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
