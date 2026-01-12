# Contributing to OSRCT Benchmark

Thank you for your interest in contributing to the OSRCT Benchmark!

## Ways to Contribute

### 1. Report Bugs
If you find a bug, please open an issue using the Bug Report template.

### 2. Submit New Methods
We welcome submissions of new causal inference methods to the benchmark leaderboard.

**Requirements:**
- Method must accept standard interface: `method(data, treatment_col, outcome_col, covariates)`
- Method must return dict with keys: `ate`, `se`, `ci_lower`, `ci_upper`
- Method should run in <60 seconds per dataset
- Include documentation and dependencies

Use the Method Submission issue template to submit your method.

### 3. Improve Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Translate documentation

### 4. Add New Datasets
If you have RCT data suitable for OSRCT transformation:
1. Data must be publicly available or shareable
2. Include proper citations
3. Follow our data format standards

## Development Setup

```bash
# Clone the repository
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Include docstrings for functions and classes
- Maximum line length: 100 characters

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit with clear message (`git commit -m "Add: description"`)
6. Push to your fork (`git push origin feature/your-feature`)
7. Open a Pull Request

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue or contact the maintainers.
