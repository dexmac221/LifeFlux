# Contributing to LifeFlux

We welcome contributions to LifeFlux! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/lifeflow.git
   cd lifeflow
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

3. **Verify Installation**
   ```bash
   pytest tests/
   python tests/benchmark.py
   ```

## Development Guidelines

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **MyPy**: Type checking  
- **Flake8**: Linting
- **Pytest**: Testing

```bash
# Format code
black src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run tests
pytest tests/ -v
```

### Code Organization

```
src/
├── lifeflux2d.py          # 2D Life implementation
├── lifeflux3d.py         # 3D Life implementation
└── __init__.py       # Package initialization

tests/
├── test_life2d.py    # 2D unit tests
├── test_lifeflux3d.py    # 3D unit tests
├── test_integration.py # Integration tests
└── benchmark.py      # Performance benchmarks

examples/
├── basic_usage.py    # Simple examples
├── advanced_demo.py  # Complex scenarios
└── pattern_gallery.py # Interesting patterns
```

### Testing Requirements

All contributions must include appropriate tests:

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test complete workflows
3. **Performance Tests**: Verify no significant performance regression
4. **CUDA Tests**: If CUDA functionality is modified

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest -m "not cuda"  # Skip CUDA tests
pytest -m "not slow"  # Skip slow tests
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and platform
- CUDA version (if applicable)
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

For new features, please:
- Describe the use case and motivation
- Provide examples of the desired API
- Consider implementation complexity
- Discuss potential breaking changes

### Code Contributions

#### Algorithm Improvements
- Performance optimizations
- New cellular automaton rules
- Advanced visualization techniques
- Memory usage optimizations

#### Platform Support
- Additional CUDA versions
- Alternative GPU backends (OpenCL, Metal)
- Distributed computing integration
- Mobile/embedded platforms

#### Visualization Enhancements
- New color mapping schemes
- Interactive controls
- Export format support
- Real-time parameter adjustment

## Pull Request Process

1. **Branch Naming**
   ```
   feature/description
   bugfix/issue-number
   performance/optimization-area
   ```

2. **Commit Messages**
   ```
   type(scope): description
   
   Examples:
   feat(life2d): add new boundary conditions
   fix(cuda): resolve memory leak in large grids
   perf(life3d): optimize neighbor counting algorithm
   docs(readme): update installation instructions
   ```

3. **PR Requirements**
   - [ ] Tests pass locally
   - [ ] New tests for added functionality
   - [ ] Documentation updates
   - [ ] Performance impact assessed
   - [ ] CUDA compatibility verified (if applicable)

4. **Review Process**
   - Automated testing on multiple platforms
   - Code quality checks
   - Performance benchmarking
   - Manual testing of new features

## Performance Considerations

### Optimization Guidelines

1. **CPU Performance**
   - Use NumPy vectorization
   - Minimize Python loops
   - Profile with `cProfile`
   - Consider Numba for hot paths

2. **GPU Performance**
   - Minimize CPU↔GPU transfers
   - Use CuPy efficiently
   - Profile with `nvprof` or Nsight
   - Consider memory coalescing

3. **Memory Management**
   - Monitor memory usage growth
   - Use appropriate data types
   - Clean up temporary arrays
   - Consider memory pooling

### Benchmarking

Before submitting performance changes:

```bash
# Baseline benchmark
git checkout main
python tests/benchmark.py > baseline.txt

# Your changes
git checkout your-branch
python tests/benchmark.py > optimized.txt

# Compare results
diff baseline.txt optimized.txt
```

## Documentation

### Code Documentation

- Use docstrings for all public functions
- Include parameter types and descriptions
- Provide usage examples
- Document performance characteristics

```python
def simulate_step(self, grid: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Execute one simulation step using cellular automaton rules.
    
    Args:
        grid: Current cell state array (H, W) of int32
        
    Returns:
        Tuple of (new_grid, convergence_metric)
        
    Performance:
        O(HW) time complexity, O(1) additional memory
        
    Example:
        >>> sim = LifeFlux2DSimulator(100, 100)
        >>> new_grid, delta = sim.simulate_step(sim.grid)
    """
```

### User Documentation

- Update README.md for new features
- Add examples to docs/ directory
- Update command-line help text
- Include performance notes

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New backward-compatible features
- **Patch** (0.0.X): Bug fixes

### Release Checklist

- [ ] All tests pass on supported platforms
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped
- [ ] Performance benchmarks are run
- [ ] Example scripts are tested

## Community

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: General questions, ideas
- **Pull Requests**: Code contributions

### Code of Conduct

We follow a simple code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Celebrate diverse perspectives

## Recognition

Contributors are recognized in:
- README.md contributors section
- CHANGELOG.md for significant changes
- Git commit history
- GitHub contributors page

Thank you for contributing to LifeFlux! 🌊✨
