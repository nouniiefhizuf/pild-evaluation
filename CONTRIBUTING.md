# Contributing to PhysBench-1K

We welcome contributions! Please follow these guidelines:

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`

## Code Style

- **Black** for formatting (line length: 100)
- **Flake8** for linting
- **Type hints** for all public functions
- **Docstrings** in Google style

## Testing

All new features must include tests:

```bash
pytest tests/test_your_feature.py -v
```

## Pull Request Process

1. Ensure tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Update documentation if needed
4. Submit PR with detailed description

## Reporting Issues

Please include:
- Python version
- Full error traceback
- Minimal reproducible example
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
