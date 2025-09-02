# Life3D - 3D Cellular Automaton Visualization

Extended from Life 2.0 into full 3D space with real-time OpenGL rendering using Open3D.

## Features

- **3D Conway's Life** with 26-neighbor rules (survive: 4-7 neighbors, birth: 6-7 neighbors)
- **3D Life 2.0** with potential field Φ and volumetric dye advection
- **CUDA acceleration** for large 3D grids using CuPy
- **Real-time 3D visualization** with OpenGL point clouds
- **Interactive controls** (when supported by Open3D version)
- **Volumetric dye field** showing flow dynamics in 3D space

## Requirements

```bash
pip install open3d numpy scipy cupy-cuda12x  # For CUDA support
# or
pip install open3d numpy scipy              # CPU only
```

## Usage

### Basic Usage

```bash
# Standard 3D Conway's Life
python lifeflux3d.py --size 32 --life-mode conway3d

# 3D Life 2.0 with potential field
python lifeflux3d.py --size 32 --life-mode life3d --phi-influence 0.5

# Large grid with CUDA acceleration
python lifeflux3d.py --size 64 --life-mode life3d --cuda

# Custom dimensions
python lifeflux3d.py --width 48 --height 32 --depth 24
```

### Command Line Parameters

- `--size N`: Set 3D grid size to N×N×N (default: 64)
- `--width W --height H --depth D`: Custom dimensions
- `--life-mode {life3d,conway3d}`: Simulation mode (default: life3d)
- `--phi-influence FLOAT`: Potential field influence 0.0-1.0 (default: 0.5)
- `--kernel-sigma FLOAT`: Gaussian blur sigma (default: 2.0)
- `--cuda`: Enable CUDA acceleration
- `--seed N`: Random seed for reproducibility

## Controls

When interactive controls are available (depends on Open3D version):

- **SPACE** - Play/Pause simulation
- **R** - Reset to random state
- **C** - Toggle cell visibility
- **D** - Toggle dye field visibility
- **P** - Toggle potential field (Φ) visibility
- **+/-** - Adjust volume rendering threshold

## 3D Rules

### Conway's Life 3D
- **Survival**: Live cell with 4-7 neighbors survives
- **Birth**: Dead cell with 6-7 neighbors becomes alive
- Uses 26-connectivity (including diagonal neighbors)

### Life 2.0 3D
- Base Conway rules + potential field influence
- Potential field Φ computed by 3D Gaussian blur of cell density
- High Φ → increased birth probability
- Low Φ → increased death probability
- Dye field advected by Φ gradient using Semi-Lagrangian method

## Visualization Modes

1. **Cells** (gray points): Live cellular automaton cells
2. **Dye Field** (colored points): Volumetric flow visualization
   - Orange: Areas of cell birth/death
   - Color intensity: Flow magnitude
3. **Phi Field** (blue points): Potential field visualization
   - Blue intensity: Field strength

## Performance Tips

- Start with smaller grids (16×16×16) for testing
- Use `--cuda` for grids larger than 32×32×32
- Adjust volume threshold with +/- to control point density
- Lower phi-influence (0.1-0.3) for more stable dynamics

## Examples

### Stable Patterns
```bash
# Conservative Life 2.0 with low chaos
python lifeflux3d.py --size 24 --life-mode life3d --phi-influence 0.2 --seed 42
```

### Chaotic Dynamics
```bash  
# High chaos with strong potential field influence
python lifeflux3d.py --size 32 --life-mode life3d --phi-influence 0.8 --cuda
```

### Pure 3D Conway's Life
```bash
# Classic 3D cellular automaton
python lifeflux3d.py --size 40 --life-mode conway3d --cuda --seed 123
```

## Technical Details

- **3D Neighbor Counting**: Uses 3×3×3 convolution kernel (26-connectivity)
- **Potential Field**: 3D Gaussian blur with configurable sigma
- **Dye Advection**: 3D Semi-Lagrangian method with trilinear interpolation
- **Visualization**: Open3D point clouds with real-time updates
- **CUDA Support**: CuPy for GPU-accelerated computation

## Troubleshooting

**"AttributeError: register_key_callback"**
- Some Open3D versions don't support interactive callbacks
- Simulation will run in auto-play mode instead

**"ValueError: operands could not be broadcast"**  
- Fixed in current version with proper dimension handling

**Performance Issues**
- Use smaller grid sizes without CUDA
- Enable CUDA for grids >32³
- Lower volume threshold to reduce rendered points

**Empty Visualization**
- Check if initial random density is too low
- Try different random seeds
- Ensure grid size is reasonable for your system

## Architecture

```
LifeFlux3DSimulator
├── 3D cellular automaton engine
├── CUDA/CPU array management  
├── 3D convolution operations
├── Volumetric dye advection
└── Convergence analysis

Life3DViewer  
├── Open3D visualization
├── Point cloud management
├── Interactive controls
└── Real-time updates
```

The 3D implementation extends all Life 2.0 concepts into volumetric space, creating rich emergent behaviors in true 3D cellular automata with flowing dye fields that reveal the invisible dynamics of the system.