# Installation Guide

This guide covers different installation methods for LifeFlux and its dependencies.

## Quick Installation

### Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/lifeflow.git
cd lifeflow

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install lifeflow
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/lifeflow.git
cd lifeflow

# Create conda environment
conda create -n lifeflow python=3.10
conda activate lifeflow

# Install dependencies
conda install numpy matplotlib scipy
pip install imageio

# Install LifeFlux
pip install -e .
```

## Dependency Details

### Core Dependencies (Required)

- **Python 3.9+**: The minimum Python version
- **NumPy ≥1.20.0**: Numerical computing foundation
- **Matplotlib ≥3.5.0**: 2D visualization and GUI
- **SciPy ≥1.8.0**: Scientific computing functions
- **imageio ≥2.20.0**: GIF export functionality

### Optional Dependencies

#### CUDA Acceleration
For GPU acceleration on NVIDIA hardware:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x  
pip install cupy-cuda11x

# Verify CUDA installation
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

#### 3D Visualization
For 3D cellular automaton visualization:

```bash
pip install open3d>=0.15.0

# Verify Open3D installation
python -c "import open3d; print(open3d.__version__)"
```

#### Development Tools
For contributing to LifeFlux:

```bash
pip install pytest pytest-cov black mypy flake8
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# System dependencies
sudo apt update
sudo apt install python3 python3-pip python3-dev

# For CUDA support (if you have NVIDIA GPU)
sudo apt install nvidia-cuda-toolkit

# Install LifeFlux
pip install lifeflow
```

### macOS

```bash
# Using Homebrew
brew install python

# For better performance, consider using conda
brew install --cask anaconda

# Install LifeFlux
pip install lifeflow
```

### Windows

```bash
# Using Anaconda (recommended)
# Download and install Anaconda from https://anaconda.com

# Open Anaconda Prompt
conda create -n lifeflow python=3.10
conda activate lifeflow
pip install lifeflow

# For CUDA support, install CUDA toolkit from NVIDIA
```

## Verification

After installation, verify everything works:

```bash
# Basic verification
python -c "import sys; sys.path.insert(0, 'src'); from life2 import LifeFlux2DSimulator; print('✓ Life2D works')"

# 3D verification (if Open3D installed)
python -c "import sys; sys.path.insert(0, 'src'); from life3d import LifeFlux3DSimulator; print('✓ Life3D works')"

# CUDA verification (if CuPy installed)
python -c "import cupy; print('✓ CUDA available')"

# Run test suite
pytest tests/
```

## Troubleshooting

### Common Issues

#### "Import Error: No module named 'cupy'"
- CUDA support is optional. Install with `pip install cupy-cuda12x`
- Or run without CUDA: `python src/lifeflux2d.py --matrix 100x100` (no --cuda flag)

#### "Import Error: No module named 'open3d'"
- 3D visualization is optional. Install with `pip install open3d`
- Or use 2D mode: `python src/lifeflux2d.py --realtime`

#### Slow performance
- Install CUDA support for GPU acceleration
- Reduce grid size for real-time interaction
- Use batch mode for large simulations

#### Display issues
- Ensure you have working X11/display forwarding for SSH
- For headless systems, use batch mode or headless 3D export
- On macOS, may need XQuartz for matplotlib

### CUDA Troubleshooting

#### Check CUDA version
```bash
nvidia-smi  # Check driver version
nvcc --version  # Check toolkit version
```

#### Install correct CuPy version
```bash
# Check CUDA version and install matching CuPy
# CUDA 12.x: pip install cupy-cuda12x
# CUDA 11.x: pip install cupy-cuda11x
```

#### Memory issues
```bash
# Check GPU memory
nvidia-smi

# Use smaller grid sizes
python src/lifeflux2d.py --matrix 200x200 --cuda  # Instead of 500x500
```

### Performance Optimization

#### For large simulations:
```bash
# Enable CUDA
python src/lifeflux2d.py --cuda --matrix 300x300

# Use batch mode (faster than real-time)
python src/lifeflux2d.py --cuda --matrix 500x500 --steps 200 --out-dir results

# Reduce visualization overhead
python src/lifeflux2d.py --cuda --matrix 400x400 --display-scale 1
```

#### For real-time visualization:
```bash
# Optimal settings for smooth interaction
python src/lifeflux2d.py --realtime --matrix 100x100 --display-scale 4

# High frame rate
python src/lifeflux2d.py --realtime --matrix 80x80 --cuda
```

## Environment Configuration

### Jupyter Notebook Support

```python
# In Jupyter notebook
%matplotlib inline
import sys
sys.path.append('path/to/lifeflow/src')

from life2 import LifeFlux2DSimulator
sim = LifeFlux2DSimulator(50, 50, seed=42)
```

### Docker Support

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -e .

CMD ["python", "examples/basic_usage.py"]
```

### Virtual Environment Best Practices

```bash
# Create isolated environment
python -m venv lifeflow-env
source lifeflow-env/bin/activate  # Linux/macOS
# lifeflow-env\Scripts\activate  # Windows

# Install with development dependencies
pip install -e .[dev]

# Freeze dependencies
pip freeze > requirements-dev.txt

# Deactivate when done
deactivate
```

## Next Steps

After successful installation:

1. **Try the examples**: `python examples/basic_usage.py`
2. **Run tests**: `pytest tests/`
3. **Interactive mode**: `python src/lifeflux2d.py --realtime`
4. **Read the guides**: Check `docs/usage_guide.md` and `docs/3d_guide.md`
5. **Experiment**: Try different parameters and create your own patterns

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Look at existing [GitHub Issues](https://github.com/yourusername/lifeflow/issues)
3. Run `python tests/benchmark.py` to diagnose performance
4. Create a new issue with your system info and error messages
