# API Reference

Complete reference for LifeFlux classes and functions.

## Life2D Module (`lifeflux2d.py`)

### LifeFlux2DSimulator Class

Main simulator class for 2D cellular automaton with flow dynamics.

```python
class LifeFlux2DSimulator:
    def __init__(self, width=100, height=100, seed=2, 
                 phi_low=0.05, phi_high=0.15, stochastic_eps=0.002,
                 flow_gain=6.0, dt_flow=0.7, kernel_radius=6, 
                 kernel_sigma=3.0, out_dir=".", display_scale=1,
                 interpolation='nearest', use_life2=True, 
                 use_cuda=False, phi_influence=0.0, colored_dye=True)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 100 | Grid width in cells |
| `height` | int | 100 | Grid height in cells |
| `seed` | int | 2 | Random seed for reproducible results |
| `phi_low` | float | 0.05 | Low potential threshold |
| `phi_high` | float | 0.15 | High potential threshold |
| `stochastic_eps` | float | 0.002 | Random flip probability |
| `flow_gain` | float | 6.0 | Flow field strength multiplier |
| `dt_flow` | float | 0.7 | Flow advection time step |
| `kernel_radius` | int | 6 | Gaussian kernel radius |
| `kernel_sigma` | float | 3.0 | Gaussian blur standard deviation |
| `out_dir` | str | "." | Output directory for GIFs |
| `display_scale` | int | 1 | Visualization scaling factor |
| `interpolation` | str | 'nearest' | Scaling interpolation method |
| `use_life2` | bool | True | Enable Life 2.0 mode vs pure Conway |
| `use_cuda` | bool | False | Enable CUDA GPU acceleration |
| `phi_influence` | float | 0.0 | Field influence on Life rules (0.0-1.0) |
| `colored_dye` | bool | True | Use colored vs grayscale dye field |

#### Methods

##### `step() -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Execute one simulation step.

**Returns:**
- `cell_rgb`: Cell visualization as RGB image (H×W×3)
- `dye_rgb`: Dye field visualization as RGB image (H×W×3)  
- `combo`: Combined visualization as RGB image (H×W×3)

**Example:**
```python
sim = LifeFlux2DSimulator(100, 100, seed=42)
cells, dye, combo = sim.step()
```

##### `reset() -> None`
Reset simulation to initial random state.

```python
sim.reset()  # New random pattern
```

##### `save_gifs() -> None`
Export recorded frames as GIF files.

```python
sim.running = True  # Enable recording
for _ in range(100):
    sim.step()
sim.save_gifs()  # Export to out_dir
```

##### `neighbor_sum(grid) -> np.ndarray`
Count neighbors for each cell (8-connectivity).

**Parameters:**
- `grid`: Cell grid as 2D array

**Returns:**
- Neighbor count array (same shape as input)

##### `blur(field, sigma) -> np.ndarray`
Apply Gaussian blur to field.

**Parameters:**
- `field`: 2D array to blur
- `sigma`: Gaussian standard deviation

**Returns:**
- Blurred field array

##### `create_colored_dye_field(dye, vx, vy) -> np.ndarray`
Generate colored visualization of dye field.

**Parameters:**
- `dye`: Dye concentration field
- `vx`, `vy`: Velocity components

**Returns:**
- RGB image array (H×W×3, uint8)

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `grid` | array | Current cell state (H×W) |
| `dye` | array | Dye concentration field (H×W) |
| `phi` | array | Potential field (H×W) |
| `vx`, `vy` | array | Flow velocity components (H×W) |
| `step_count` | int | Current simulation step |
| `is_converged` | bool | Convergence status |
| `current_delta` | float | Current change rate |

### RealtimeViewer Class

Interactive matplotlib-based visualization.

```python
class RealtimeViewer:
    def __init__(self, simulator: LifeFlux2DSimulator)
```

#### Methods

##### `show() -> None`
Start interactive visualization window.

```python
sim = LifeFlux2DSimulator(100, 100, seed=42)
viewer = RealtimeViewer(sim)
viewer.show()  # Opens GUI window
```

## Life3D Module (`lifeflux3d.py`)

### LifeFlux3DSimulator Class

3D cellular automaton with volumetric visualization.

```python
class LifeFlux3DSimulator:
    def __init__(self, width=64, height=64, depth=64, 
                 use_life3d=True, phi_influence=0.5, 
                 kernel_sigma=2.0, use_cuda=False, seed=None)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 64 | Grid width |
| `height` | int | 64 | Grid height |  
| `depth` | int | 64 | Grid depth |
| `use_life3d` | bool | True | Life3D vs Conway3D mode |
| `phi_influence` | float | 0.5 | Field influence strength |
| `kernel_sigma` | float | 2.0 | 3D Gaussian blur sigma |
| `use_cuda` | bool | False | GPU acceleration |
| `seed` | int | None | Random seed |

#### Methods

##### `step() -> None`
Execute one 3D simulation step.

```python
sim = LifeFlux3DSimulator(32, 32, 32)
sim.step()
```

##### `count_3d_neighbors(cells) -> np.ndarray`
Count 3D neighbors (26-connectivity).

**Parameters:**
- `cells`: 3D boolean array

**Returns:**
- Neighbor count array (D×H×W)

##### `update_potential_field() -> None`
Update 3D potential field from cell density.

##### `export_ply(filename) -> None`
Export current state as PLY point cloud.

**Parameters:**
- `filename`: Output PLY file path

```python
sim.export_ply("cells_step_0100.ply")
```

### Life3DViewer Class

3D visualization using Open3D.

```python
class Life3DViewer:
    def __init__(self, simulator: LifeFlux3DSimulator)
```

#### Methods

##### `run() -> None`
Start interactive 3D visualization.

```python
sim = LifeFlux3DSimulator(32, 32, 32)
viewer = Life3DViewer(sim)
viewer.run()  # Opens 3D window
```

## Utility Functions

### Command Line Interface

Both modules provide command-line interfaces:

```bash
# 2D Life
python src/lifeflux2d.py --help
python src/lifeflux2d.py --realtime --matrix 100x100 --cuda

# 3D Life  
python src/lifeflux3d.py --help
python src/lifeflux3d.py --size 32 --life-mode life3d --cuda
```

### Array Utilities

#### `to_cpu(array) -> np.ndarray`
Convert CuPy GPU array to NumPy CPU array.

```python
if sim.use_cuda:
    cpu_array = sim.to_cpu(sim.grid)
else:
    cpu_array = sim.grid
```

#### `scale_image(img) -> np.ndarray`
Scale image by display_scale factor.

```python
scaled = sim.scale_image(small_image)
```

## Configuration Examples

### High Performance Setup

```python
# Large CUDA-accelerated simulation
sim = LifeFlux2DSimulator(
    width=500, height=500,
    use_cuda=True,
    use_life2=True,
    phi_influence=0.3,
    flow_gain=10.0,
    display_scale=1,  # Minimal visualization overhead
    out_dir="results"
)
```

### Interactive Visualization

```python
# Smooth real-time interaction
sim = LifeFlux2DSimulator(
    width=80, height=80,
    use_life2=True,
    phi_influence=0.2,
    display_scale=4,  # High-resolution display
    colored_dye=True
)
viewer = RealtimeViewer(sim)
viewer.show()
```

### Scientific Analysis

```python
# Convergence analysis setup
sim = LifeFlux2DSimulator(
    width=100, height=100,
    use_life2=False,  # Pure Conway for comparison
    seed=42,  # Reproducible results
    stochastic_eps=0.0  # Deterministic evolution
)

# Track convergence
for step in range(1000):
    sim.step()
    if sim.is_converged:
        print(f"Converged at step {step}")
        break
```

### 3D Volumetric Analysis

```python  
# Large 3D system with field dynamics
sim = LifeFlux3DSimulator(
    width=64, height=64, depth=64,
    use_life3d=True,
    phi_influence=0.4,
    kernel_sigma=3.0,
    use_cuda=True
)

# Export time series
for step in range(0, 100, 10):
    for _ in range(10):
        sim.step()
    sim.export_ply(f"3d_step_{step:04d}.ply")
```

## Error Handling

### Common Exceptions

#### `ImportError: No module named 'cupy'`
CUDA not available. Set `use_cuda=False` or install CuPy.

#### `RuntimeError: CUDA out of memory`
Reduce grid size or increase GPU memory.

#### `ValueError: operands could not be broadcast`
Array shape mismatch. Check grid dimensions.

### Debugging Tips

```python
# Check array shapes
print(f"Grid: {sim.grid.shape}")
print(f"Phi: {sim.phi.shape}")
print(f"Flow: {sim.vx.shape}, {sim.vy.shape}")

# Monitor memory usage
if sim.use_cuda:
    import cupy
    mempool = cupy.get_default_memory_pool()
    print(f"GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")

# Performance profiling
import time
start = time.time()
for _ in range(100):
    sim.step()
fps = 100 / (time.time() - start)
print(f"Performance: {fps:.1f} FPS")
```
