#!/usr/bin/env python3
"""
Advanced demonstration of LifeFlow capabilities

This script showcases advanced features including:
- Complex parameter combinations
- Performance optimization techniques  
- Advanced visualization methods
- Scientific analysis workflows
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

def demo_chaos_emergence():
    """Demonstrate emergence of chaos with increasing phi influence"""
    print("🌊 Chaos Emergence Analysis")
    print("-" * 50)
    
    from lifeflux2d import LifeFlux2DSimulator
    
    phi_values = np.linspace(0.0, 1.0, 11)
    chaos_metrics = []
    
    for phi in phi_values:
        sim = LifeFlux2DSimulator(
            width=80, height=80,
            use_life2=True,
            phi_influence=phi,
            flow_gain=10.0,
            seed=42
        )
        
        # Measure system dynamics
        deltas = []
        for step in range(60):
            sim.step()
            if step > 10:  # Skip initial transient
                deltas.append(sim.current_delta)
        
        avg_delta = np.mean(deltas[-20:]) if deltas else 0
        chaos_metrics.append(avg_delta)
        
        status = "chaotic" if avg_delta > 0.01 else "stable"
        print(f"φ={phi:.1f}: Δ={avg_delta:.4f} ({status})")
    
    # Find chaos transition point
    chaos_threshold = 0.01
    transition_idx = next((i for i, delta in enumerate(chaos_metrics) 
                          if delta > chaos_threshold), len(chaos_metrics))
    
    if transition_idx < len(phi_values):
        print(f"\n🎯 Chaos transition at φ ≈ {phi_values[transition_idx]:.1f}")
    else:
        print(f"\n📊 System remains stable across all φ values tested")
    
    return phi_values, chaos_metrics

def demo_attractor_analysis():
    """Analyze attractor formation in Life 2.0"""
    print("\n🎯 Attractor Formation Analysis") 
    print("-" * 50)
    
    from lifeflux2d import LifeFlux2DSimulator
    
    # Test different flow gains for attractor strength
    flow_gains = [2.0, 6.0, 12.0, 20.0]
    attractor_data = []
    
    for flow_gain in flow_gains:
        sim = LifeFlux2DSimulator(
            width=60, height=60,
            use_life2=True,
            phi_influence=0.4,
            flow_gain=flow_gain,
            kernel_sigma=4.0,
            seed=123
        )
        
        # Run simulation to develop attractors
        for _ in range(40):
            sim.step()
        
        # Analyze flow field
        vx_cpu = sim.to_cpu(sim.vx)
        vy_cpu = sim.to_cpu(sim.vy)
        
        flow_magnitude = np.sqrt(vx_cpu**2 + vy_cpu**2)
        max_flow = np.max(flow_magnitude)
        avg_flow = np.mean(flow_magnitude)
        
        # Calculate vorticity (curl of velocity field)
        dvx_dy = np.gradient(vx_cpu, axis=0)  
        dvy_dx = np.gradient(vy_cpu, axis=1)
        vorticity = dvy_dx - dvx_dy
        max_vorticity = np.max(np.abs(vorticity))
        
        attractor_data.append({
            'flow_gain': flow_gain,
            'max_flow': max_flow,
            'avg_flow': avg_flow,
            'max_vorticity': max_vorticity
        })
        
        print(f"Flow gain {flow_gain:4.1f}: max_flow={max_flow:.3f}, "
              f"vorticity={max_vorticity:.3f}")
    
    # Find optimal flow gain for interesting dynamics
    optimal_idx = np.argmax([d['max_vorticity'] for d in attractor_data])
    optimal_gain = attractor_data[optimal_idx]['flow_gain']
    print(f"\n🌀 Optimal flow gain for vortex formation: {optimal_gain}")
    
    return attractor_data

def demo_scale_performance():
    """Demonstrate performance scaling with grid size"""
    print("\n⚡ Performance Scaling Analysis")
    print("-" * 50)
    
    from lifeflux2d import LifeFlux2DSimulator
    
    # Test different grid sizes
    sizes = [50, 100, 200, 300]
    performance_data = []
    
    for size in sizes:
        print(f"\nTesting {size}x{size} grid...")
        
        # CPU performance
        start_time = time.time()
        sim_cpu = LifeFlux2DSimulator(width=size, height=size, use_cuda=False, seed=42)
        
        for _ in range(5):
            sim_cpu.step()
        
        cpu_time = time.time() - start_time
        cpu_fps = 5 / cpu_time
        
        data_point = {
            'size': size,
            'cpu_fps': cpu_fps,
            'memory_mb': size * size * 20 / 1024 / 1024  # Rough estimate
        }
        
        # CUDA performance (if available)
        try:
            import cupy
            start_time = time.time()
            sim_cuda = LifeFlux2DSimulator(width=size, height=size, use_cuda=True, seed=42)
            
            for _ in range(5):
                sim_cuda.step()
            
            cuda_time = time.time() - start_time
            cuda_fps = 5 / cuda_time
            speedup = cuda_fps / cpu_fps if cpu_fps > 0 else float('inf')
            
            data_point['cuda_fps'] = cuda_fps
            data_point['speedup'] = speedup
            
            print(f"  CPU:  {cpu_fps:5.1f} FPS")
            print(f"  CUDA: {cuda_fps:5.1f} FPS ({speedup:.1f}x speedup)")
            
        except ImportError:
            data_point['cuda_fps'] = None
            data_point['speedup'] = None
            print(f"  CPU:  {cpu_fps:5.1f} FPS")
            print(f"  CUDA: Not available")
        
        performance_data.append(data_point)
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("Size  | CPU FPS | CUDA FPS | Speedup | Memory")
    print("-" * 45)
    for data in performance_data:
        cuda_str = f"{data['cuda_fps']:7.1f}" if data['cuda_fps'] else "   N/A"
        speedup_str = f"{data['speedup']:6.1f}x" if data['speedup'] else "   N/A"
        print(f"{data['size']:4d}  | {data['cpu_fps']:7.1f} | {cuda_str} | {speedup_str} | "
              f"{data['memory_mb']:5.1f}MB")
    
    return performance_data

def demo_convergence_patterns():
    """Analyze convergence behavior for different patterns"""
    print("\n🎯 Convergence Pattern Analysis")
    print("-" * 50)
    
    from lifeflux2d import LifeFlux2DSimulator
    
    # Test different initial patterns
    patterns = {
        'random': lambda w, h: np.random.random((h, w)) < 0.15,
        'sparse': lambda w, h: np.random.random((h, w)) < 0.05,  
        'dense': lambda w, h: np.random.random((h, w)) < 0.25,
        'structured': lambda w, h: create_structured_pattern(w, h)
    }
    
    convergence_results = {}
    
    for pattern_name, pattern_func in patterns.items():
        print(f"\nTesting '{pattern_name}' pattern...")
        
        sim = LifeFlux2DSimulator(width=50, height=50, use_life2=True, 
                           phi_influence=0.2, seed=42)
        
        # Set custom initial pattern
        initial_pattern = pattern_func(50, 50).astype(np.int32)
        if sim.use_cuda:
            import cupy as cp
            sim.grid = cp.asarray(initial_pattern)
        else:
            sim.grid = initial_pattern
        
        # Track convergence
        convergence_step = None
        deltas = []
        
        for step in range(100):
            sim.step()
            deltas.append(sim.current_delta)
            
            if sim.is_converged and convergence_step is None:
                convergence_step = step
        
        final_cells = np.sum(sim.to_cpu(sim.grid))
        avg_delta_final = np.mean(deltas[-10:]) if len(deltas) >= 10 else sim.current_delta
        
        convergence_results[pattern_name] = {
            'convergence_step': convergence_step,
            'final_cells': final_cells,
            'final_delta': avg_delta_final,
            'converged': sim.is_converged
        }
        
        status = f"step {convergence_step}" if convergence_step else "no convergence"
        print(f"  Result: {final_cells} cells, Δ={avg_delta_final:.5f}, {status}")
    
    return convergence_results

def create_structured_pattern(width: int, height: int) -> np.ndarray:
    """Create a structured initial pattern with known properties"""
    pattern = np.zeros((height, width), dtype=bool)
    
    # Add some gliders
    glider_positions = [(10, 10), (15, 20), (25, 30)]
    
    for y, x in glider_positions:
        if y + 3 < height and x + 3 < width:
            # Standard glider pattern
            pattern[y, x+1] = True
            pattern[y+1, x+2] = True 
            pattern[y+2, x:x+3] = True
    
    # Add some blocks (still lifes)
    block_positions = [(5, 40), (35, 5), (40, 40)]
    
    for y, x in block_positions:
        if y + 2 < height and x + 2 < width:
            pattern[y:y+2, x:x+2] = True
    
    return pattern

def demo_3d_capabilities():
    """Demonstrate 3D cellular automaton capabilities"""
    print("\n🧊 3D Cellular Automaton Demo")
    print("-" * 50)
    
    try:
        from lifeflux3d import LifeFlux3DSimulator
        
        print("Testing 3D Life simulation...")
        
        # Create 3D simulator
        sim3d = LifeFlux3DSimulator(
            width=24, height=24, depth=24,
            use_life3d=True,
            phi_influence=0.3,
            seed=42
        )
        
        initial_cells = np.sum(sim3d.to_cpu_array(sim3d.cells))
        print(f"Initial 3D cells: {initial_cells}")
        
        # Run 3D simulation
        for step in range(10):
            sim3d.step()
            
            if step % 3 == 0:
                live_cells = np.sum(sim3d.to_cpu_array(sim3d.cells))
                print(f"Step {step:2d}: {live_cells:4d} live cells")
        
        final_cells = np.sum(sim3d.to_cpu_array(sim3d.cells))
        print(f"\n3D simulation: {initial_cells} → {final_cells} cells")
        
        # Test 3D potential field
        phi_3d = sim3d.to_cpu_array(sim3d.phi)
        phi_max = np.max(phi_3d)
        phi_mean = np.mean(phi_3d)
        
        print(f"3D potential field: max={phi_max:.4f}, mean={phi_mean:.4f}")
        print("✅ 3D capabilities verified")
        
    except ImportError:
        print("⚠️  3D capabilities not available (missing dependencies)")
    except Exception as e:
        print(f"❌ 3D test failed: {e}")

def demo_advanced_visualization():
    """Demonstrate advanced visualization techniques"""
    print("\n🎨 Advanced Visualization Demo")
    print("-" * 50)
    
    from lifeflux2d import LifeFlux2DSimulator
    
    # Create simulator with high-resolution output
    sim = LifeFlux2DSimulator(
        width=60, height=60,
        use_life2=True,
        phi_influence=0.5,
        flow_gain=12.0,
        colored_dye=True,
        display_scale=2,  # Higher resolution
        seed=999
    )
    
    print("Generating high-resolution visualization data...")
    
    # Run simulation to develop interesting patterns
    for step in range(25):
        cell_rgb, dye_rgb, combo = sim.step()
        
        if step % 10 == 0:
            print(f"  Step {step}: Generated {cell_rgb.shape} image")
    
    # Analyze color distribution in dye field
    dye_cpu = sim.to_cpu(sim.dye)
    vx_cpu = sim.to_cpu(sim.vx) 
    vy_cpu = sim.to_cpu(sim.vy)
    
    # Create final colored visualization
    colored_field = sim.create_colored_dye_field(sim.dye, sim.vx, sim.vy)
    
    # Color analysis
    unique_colors = len(np.unique(colored_field.reshape(-1, 3), axis=0))
    flow_vectors = np.sum(vx_cpu**2 + vy_cpu**2 > 0.01)
    
    print(f"Visualization analysis:")
    print(f"  Unique colors: {unique_colors}")
    print(f"  Active flow vectors: {flow_vectors}")
    print(f"  Output resolution: {colored_field.shape}")
    
    return colored_field

def main():
    """Run advanced demonstration"""
    print("LifeFlow Advanced Capabilities Demo")
    print("=" * 60)
    
    results = {}
    
    try:
        # Run all demos
        results['chaos'] = demo_chaos_emergence()
        results['attractors'] = demo_attractor_analysis()
        results['performance'] = demo_scale_performance()
        results['convergence'] = demo_convergence_patterns()
        demo_3d_capabilities()
        results['visualization'] = demo_advanced_visualization()
        
        print("\n" + "=" * 60)
        print("🎉 Advanced Demo Completed Successfully!")
        
        # Summary insights
        print("\n📊 Key Insights:")
        phi_values, chaos_metrics = results['chaos']
        stable_count = sum(1 for delta in chaos_metrics if delta < 0.01)
        print(f"  • {stable_count}/{len(phi_values)} φ values produce stable patterns")
        
        perf_data = results['performance']
        if any(d['cuda_fps'] for d in perf_data):
            max_speedup = max(d['speedup'] for d in perf_data if d['speedup'])
            print(f"  • Maximum CUDA speedup: {max_speedup:.1f}x")
        
        conv_data = results['convergence']
        converged_patterns = sum(1 for r in conv_data.values() if r['converged'])
        print(f"  • {converged_patterns}/{len(conv_data)} pattern types converged")
        
        print("\n🔬 Scientific Applications:")
        print("  • Chaos theory and phase transitions")
        print("  • Pattern formation and self-organization")
        print("  • Fluid dynamics visualization")
        print("  • Complex systems modeling")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nThis might be due to missing dependencies or system limitations.")
        print("Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
