# Contributing to OSRCT Benchmark

Thank you for your interest in contributing to the OSRCT Benchmark project!

## Ways to Contribute

### 1. Report Bugs

Open an issue with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, package versions)

### 2. Submit New Causal Methods

We welcome implementations of new causal inference methods:

1. Fork the repository
2. Implement your method following the standard interface:

```python
def estimate_your_method(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None
) -> Dict[str, Any]:
    """
    Your method description.

    Returns
    -------
    dict with keys: 'method', 'ate', 'se', 'ci_lower', 'ci_upper'
    """
    # Implementation
    return {
        'method': 'your_method_name',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se
    }
```

3. Add tests
4. Run benchmark evaluation
5. Submit pull request with results

### 3. Improve Documentation

- Fix typos
- Add examples
- Improve explanations
- Translate documentation

### 4. Add New Datasets

To add a new RCT dataset for OSRCT benchmarking:

1. Ensure the dataset has:
   - Binary or multi-valued treatment
   - Continuous outcome
   - Pre-treatment covariates
   - Known ground-truth ATE

2. Create preprocessing script
3. Add to dataset catalog
4. Document the data source

## Development Setup

```bash
# Clone the repository
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 isort mypy
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Format with `black`
- Sort imports with `isort`

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests and linting
5. Commit with descriptive message
6. Push to your fork
7. Open a pull request

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the work, not the person
- Report unacceptable behavior to maintainers

## Questions?

Open an issue with the "question" label or reach out to maintainers.

Thank you for contributing!
