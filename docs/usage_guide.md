# Life 2.0 — Vortices & Attractors in a Conway-like World

**Life 2.0** is a cellular automaton inspired by Conway's Game of Life, augmented with a **continuous potential field** and a **colored dye tracer** that reveals **attractors, swirls, and orbital-like patterns** — a true "probability collapse spectacle"! 🌊✨

It keeps Life's local rules but biases them with a smoothed "gravity" field Φ derived from living cells. A dye field is advected by curl-like flow from ∇Φ to reveal **eddies/mulinelli** (probability basins) with stunning colored visualizations showing flow direction and intensity.

---

## 🌟 Features

* **Pure Conway's Life** + optional **potential field bias** (tunable chaos level)
* **Colored dye tracer** showing flow direction (hue) and intensity (saturation) 🎨  
* **Real-time visualization** with speed controls and convergence metrics
* **CUDA acceleration** for giant grids (500x500+ possible with GPU)
* **Optimized performance** with FFT-based blur and GPU-optimized advection
* **Parametric resolution** scaling for detailed visualization
* **Multiple export formats** (cells, dye field, combo GIFs)

---

## 🚀 Quick Start

### Requirements

* Python 3.9+
* Packages: `numpy`, `matplotlib`, `imageio`
* Optional: `cupy` for CUDA acceleration

```bash
pip install numpy matplotlib imageio
# For CUDA acceleration:
pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11.x
```

### Basic Usage

```bash
# Realtime colored visualization
python lifeflux2d.py --realtime --matrix 100x100

# Pure Conway's Life with dye field
python lifeflux2d.py --realtime --phi-influence 0.0 --matrix 80x80

# CUDA-accelerated giant grid
python lifeflux2d.py --realtime --cuda --matrix 300x300

# Export GIFs
python lifeflux2d.py --matrix 120x120 --steps 200 --out-dir results
```

---

## 🎛️ Parameters Reference

### Grid & Simulation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--matrix WxH` | 100x100 | Grid dimensions (e.g., `200x150`) |
| `--width W` | 100 | Grid width |
| `--height H` | 100 | Grid height |
| `--seed N` | 2 | Random seed for reproducible patterns |
| `--steps N` | 200 | Number of simulation steps |

### Life Rules & Dynamics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--life-mode MODE` | life2 | `standard` (pure Conway) or `life2` (with fields) |
| `--phi-influence X` | 0.0 | Field influence: `0.0`=pure Conway, `0.2`=gentle bias, `1.0`=full chaos |
| `--phi-low X` | 0.05 | Low potential threshold |
| `--phi-high X` | 0.15 | High potential threshold |
| `--stochastic-eps X` | 0.002 | Random flip probability |

### Flow & Visualization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--flow-gain X` | 6.0 | Flow field strength (higher = more intense vortices) |
| `--dt-flow X` | 0.7 | Flow time step |
| `--kernel-sigma X` | 3.0 | Gaussian blur width for potential field |
| `--colored-dye` | True | Colored dye field (default) |
| `--grayscale-dye` | False | Use grayscale dye instead of colored |

### Performance & Display

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cuda` | False | Enable CUDA acceleration |
| `--display-scale N` | 1 (batch), 4 (realtime) | Resolution scaling factor |
| `--realtime` | False | Interactive real-time mode |
| `--out-dir PATH` | . | Output directory for GIFs |

---

## 🌈 Color Interpretation

### Colored Dye Field Visualization
- **🔴 Red/Orange**: Flow moving right/down-right
- **🟡 Yellow**: Flow moving down  
- **🟢 Green**: Flow moving left/down-left
- **🔵 Blue/Cyan**: Flow moving up/up-left
- **🟣 Purple**: Flow moving up-right

**Brightness** = Dye concentration  
**Saturation** = Flow speed/intensity

---

## 📖 Usage Examples

### Basic Exploration

```bash
# Start with pure Conway's Life + colored flow visualization
python lifeflux2d.py --realtime --phi-influence 0.0 --matrix 100x100

# Add gentle field bias for more dynamic patterns  
python lifeflux2d.py --realtime --phi-influence 0.2 --matrix 100x100

# Full Life 2.0 chaos
python lifeflux2d.py --realtime --phi-influence 1.0 --matrix 100x100
```

### Performance & Scale

```bash
# Small detailed view
python lifeflux2d.py --realtime --matrix 60x60 --display-scale 8

# CUDA-accelerated giant grid
python lifeflux2d.py --realtime --cuda --matrix 400x400 --display-scale 2

# Massive computation
python lifeflux2d.py --cuda --matrix 800x800 --steps 500 --out-dir massive_run
```

### Flow Dynamics

```bash
# Gentle swirls
python lifeflux2d.py --realtime --flow-gain 4 --matrix 120x120

# Intense vortices  
python lifeflux2d.py --realtime --flow-gain 15 --matrix 120x120

# Extreme turbulence
python lifeflux2d.py --realtime --flow-gain 25 --matrix 120x120
```

### Reproducible Experiments

```bash
# Different seeds create different attractor patterns
python lifeflux2d.py --realtime --seed 1 --matrix 150x150   # Pattern A
python lifeflux2d.py --realtime --seed 42 --matrix 150x150  # Pattern B  
python lifeflux2d.py --realtime --seed 123 --matrix 150x150 # Pattern C

# Export specific seed for analysis
python lifeflux2d.py --cuda --seed 42 --matrix 200x200 --steps 300 --out-dir seed42_analysis
```

### Visualization Modes

```bash
# Colored dye field (default) - shows flow direction
python lifeflux2d.py --realtime --matrix 100x100

# Grayscale dye field (classic)
python lifeflux2d.py --realtime --grayscale-dye --matrix 100x100

# High resolution for detailed analysis
python lifeflux2d.py --realtime --matrix 80x80 --display-scale 6
```

---

## 🎮 Interactive Controls (Realtime Mode)

### Buttons
- **Play/Pause**: Start/stop simulation
- **Reset**: New random initial state  
- **Record**: Toggle GIF frame recording
- **Save GIFs**: Export recorded frames
- **+/-**: Speed controls (10ms to 1000ms per step)

### Display
- **Step counter**: Current simulation step
- **Recording status**: Shows if recording frames
- **Speed**: Current animation interval
- **Convergence**: Δ=change rate, converged status

---

## 📊 Understanding Convergence

The system tracks pattern stability with a 50-step sliding window:

- **Δ < 0.002**: System is converged (stable patterns)
- **Δ > 0.01**: System is chaotic/evolving
- **"converged: True"**: Pattern has stabilized

**Tuning for convergence:**
- `--phi-influence 0.0`: Always converges (pure Conway)
- `--phi-influence 0.1-0.3`: Usually converges with interesting dynamics
- `--phi-influence 0.5+`: May stay chaotic (beautiful but unstable)

---

## 🔬 Scientific Applications

### Attractor Analysis
```bash
# Study attractor formation at different scales
python lifeflux2d.py --cuda --seed 42 --matrix 200x200 --phi-influence 0.2 --steps 1000

# Compare convergence rates
python lifeflux2d.py --phi-influence 0.0 --steps 500  # Fast convergence
python lifeflux2d.py --phi-influence 0.3 --steps 500  # Slower convergence
```

### Flow Pattern Studies  
```bash
# Vortex dynamics at different flow strengths
for gain in 5 10 15 20; do
  python lifeflux2d.py --flow-gain $gain --matrix 150x150 --steps 200 --out-dir flow_$gain
done
```

### Parameter Space Exploration
```bash
# Systematic parameter sweep
python lifeflux2d.py --phi-influence 0.1 --flow-gain 8 --seed 1 --matrix 120x120 --steps 300
python lifeflux2d.py --phi-influence 0.2 --flow-gain 8 --seed 1 --matrix 120x120 --steps 300  
python lifeflux2d.py --phi-influence 0.3 --flow-gain 8 --seed 1 --matrix 120x120 --steps 300
```

---

## 🏆 Performance Tips

### For Large Grids (200x200+)
```bash
# Always use CUDA
python lifeflux2d.py --cuda --matrix 400x400

# Reduce display scale for smooth realtime
python lifeflux2d.py --realtime --cuda --matrix 300x300 --display-scale 1

# Use batch mode for massive runs
python lifeflux2d.py --cuda --matrix 800x800 --steps 1000 --out-dir massive
```

### For Smooth Visualization
```bash
# High resolution small grids
python lifeflux2d.py --realtime --matrix 60x60 --display-scale 8

# Balanced performance/quality
python lifeflux2d.py --realtime --cuda --matrix 150x150 --display-scale 3
```

---

## 🎯 Recipe Gallery

### Classic Patterns
```bash
# Pure Conway's Life with flow visualization
python lifeflux2d.py --realtime --phi-influence 0.0 --matrix 100x100

# Stable "planetary" systems  
python lifeflux2d.py --realtime --phi-influence 0.1 --phi-high 0.2 --stochastic-eps 0.001
```

### Dynamic Systems
```bash
# Gentle probability bias
python lifeflux2d.py --realtime --phi-influence 0.2 --flow-gain 8 --matrix 120x120

# Turbulent vortex fields
python lifeflux2d.py --realtime --phi-influence 0.4 --flow-gain 18 --matrix 150x150
```

### Extreme Dynamics
```bash
# Probability collapse spectacle
python lifeflux2d.py --realtime --phi-influence 0.8 --flow-gain 25 --matrix 200x200

# Chaotic attractor soup
python lifeflux2d.py --realtime --phi-influence 1.0 --flow-gain 30 --stochastic-eps 0.01
```

---

## 📁 Output Files

When using `--out-dir`, the system generates:

- `life2_cells.gif`: Binary Life evolution (white=alive, black=dead)
- `life2_dye.gif`: Colored dye field showing flow patterns  
- `life2_combo.gif`: Overlay (dye background + highlighted cells)

---

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check GPU availability
nvidia-smi

# Install correct CuPy version
pip install cupy-cuda12x  # for CUDA 12.x
pip install cupy-cuda11x  # for CUDA 11.x
```

### Performance Issues
```bash
# Reduce grid size
python lifeflux2d.py --matrix 80x80 instead of --matrix 200x200

# Lower display scale
python lifeflux2d.py --display-scale 1 instead of --display-scale 4

# Use grayscale dye (faster)
python lifeflux2d.py --grayscale-dye
```

---

## 🔬 Technical Details

### Optimizations Implemented
- **FFT-based Gaussian blur** (CPU) and `cupyx.scipy` filters (GPU)
- **Linear-index bilinear sampling** for efficient GPU advection  
- **Vectorized HSV→RGB conversion** for colored visualization
- **Potential field inertia** with smoothing factor α=0.2 for stability
- **CPU↔GPU transfers minimized** to visualization only

### Algorithm
1. **Cellular Step**: Conway's Life rules ± optional field bias
2. **Field Update**: Φ ← (1-α)Φ + α·blur(cells) with inertia  
3. **Flow Computation**: (vₓ,vᵧ) = ∇Φ × flow_gain (rotational flow)
4. **Dye Advection**: Semi-Lagrangian advection of dye field
5. **Visualization**: HSV→RGB mapping (hue=direction, sat=speed, val=concentration)

---

## 📜 License & Credits

Inspired by Conway's Game of Life with fluid dynamics extensions.  
Implements advanced optimization techniques for real-time large-scale simulation.

**Key Features:**
- ✅ CUDA GPU acceleration  
- ✅ Optimized algorithms (FFT blur, vectorized operations)
- ✅ Colored flow visualization  
- ✅ Real-time interaction with speed controls
- ✅ Convergence analysis
- ✅ Tunable chaos levels

---

**Enjoy exploring the probability collapse spectacle!** 🌊✨🎨

*Try different seeds and watch how the same rules create completely different attractor landscapes...*