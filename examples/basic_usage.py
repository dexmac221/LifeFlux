#!/usr/bin/env python3
"""
Basic usage examples for LifeFlow

This script demonstrates the fundamental features of LifeFlow
with simple, easy-to-understand examples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lifeflux2d import LifeFlux2DSimulator
import numpy as np
import matplotlib.pyplot as plt

def example_1_basic_conway():
    """Example 1: Basic Conway's Game of Life"""
    print("Example 1: Basic Conway's Game of Life")
    print("-" * 40)
    
    # Create a small Conway's Life simulator
    sim = LifeFlux2DSimulator(
        width=50, 
        height=50,
        use_life2=False,  # Pure Conway's Life
        seed=42
    )
    
    print(f"Initial live cells: {np.sum(sim.to_cpu(sim.grid))}")
    
    # Run simulation for 20 steps
    for step in range(20):
        sim.step()
        if step % 5 == 0:
            live_cells = np.sum(sim.to_cpu(sim.grid))
            print(f"Step {step:2d}: {live_cells} live cells")
    
    print(f"Final convergence status: {sim.is_converged}")
    print()

def example_2_life2_with_flow():
    """Example 2: Life 2.0 with potential field and flow visualization"""
    print("Example 2: Life 2.0 with Flow Dynamics")
    print("-" * 40)
    
    # Create Life 2.0 simulator with moderate field influence
    sim = LifeFlux2DSimulator(
        width=60,
        height=60, 
        use_life2=True,
        phi_influence=0.3,  # Moderate field influence
        flow_gain=8.0,      # Moderate flow strength
        colored_dye=True,   # Enable colored flow visualization
        seed=123
    )
    
    print(f"Configuration: {sim.width}x{sim.height} grid")
    print(f"Phi influence: {sim.phi_influence}")
    print(f"Flow gain: {sim.flow_gain}")
    
    # Run simulation and track dynamics
    for step in range(30):
        sim.step()
        
        if step % 10 == 0:
            live_cells = np.sum(sim.to_cpu(sim.grid))
            phi_max = np.max(sim.to_cpu(sim.phi))
            flow_speed = np.sqrt(sim.to_cpu(sim.vx)**2 + sim.to_cpu(sim.vy)**2)
            flow_max = np.max(flow_speed)
            
            print(f"Step {step:2d}: {live_cells} cells, φ_max={phi_max:.3f}, flow_max={flow_max:.3f}")
    
    print(f"Convergence: Δ={sim.current_delta:.5f}, converged={sim.is_converged}")
    print()

def example_3_cuda_performance():
    """Example 3: CUDA acceleration demonstration"""
    print("Example 3: CUDA Performance Comparison")
    print("-" * 40)
    
    import time
    
    # Test configuration
    size = (100, 100)
    steps = 10
    
    # CPU version
    print("Testing CPU performance...")
    start_time = time.time()
    
    sim_cpu = LifeFlux2DSimulator(
        width=size[0], 
        height=size[1],
        use_cuda=False,
        use_life2=True,
        seed=42
    )
    
    for _ in range(steps):
        sim_cpu.step()
    
    cpu_time = time.time() - start_time
    print(f"CPU: {steps} steps in {cpu_time:.3f}s ({steps/cpu_time:.1f} steps/sec)")
    
    # CUDA version (if available)
    try:
        import cupy
        print("Testing CUDA performance...")
        start_time = time.time()
        
        sim_cuda = LifeFlux2DSimulator(
            width=size[0],
            height=size[1], 
            use_cuda=True,
            use_life2=True,
            seed=42
        )
        
        for _ in range(steps):
            sim_cuda.step()
        
        cuda_time = time.time() - start_time
        speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
        
        print(f"CUDA: {steps} steps in {cuda_time:.3f}s ({steps/cuda_time:.1f} steps/sec)")
        print(f"Speedup: {speedup:.1f}x")
        
    except ImportError:
        print("CUDA not available - install CuPy for GPU acceleration")
    
    print()

def example_4_visualization_modes():
    """Example 4: Different visualization and export modes"""
    print("Example 4: Visualization and Export")
    print("-" * 40)
    
    # Create simulator with recording enabled
    sim = LifeFlux2DSimulator(
        width=40,
        height=40,
        use_life2=True,
        phi_influence=0.4,
        colored_dye=True,
        out_dir="example_output",
        seed=456
    )
    
    # Enable frame recording
    sim.running = True
    
    print("Recording 15 frames...")
    for step in range(15):
        cell_rgb, dye_rgb, combo = sim.step()
        
        if step % 5 == 0:
            print(f"  Frame {step+1}: {cell_rgb.shape} pixels")
    
    # Save GIF files
    print("Saving GIF files...")
    sim.save_gifs()
    print(f"Generated files in '{sim.out_dir}/':")
    
    import glob
    gif_files = glob.glob(f"{sim.out_dir}/*.gif")
    for gif_file in gif_files:
        file_size = os.path.getsize(gif_file) / 1024
        print(f"  {os.path.basename(gif_file)}: {file_size:.1f} KB")
    
    print()

def example_5_parameter_exploration():
    """Example 5: Parameter space exploration"""
    print("Example 5: Parameter Space Exploration")
    print("-" * 40)
    
    # Test different phi_influence values
    phi_values = [0.0, 0.2, 0.5, 0.8]
    results = []
    
    for phi in phi_values:
        sim = LifeFlux2DSimulator(
            width=30,
            height=30,
            use_life2=True,
            phi_influence=phi,
            seed=789
        )
        
        # Run for convergence analysis
        initial_cells = np.sum(sim.to_cpu(sim.grid))
        
        for step in range(50):
            sim.step()
        
        final_cells = np.sum(sim.to_cpu(sim.grid))
        stability = "converged" if sim.is_converged else "evolving"
        
        results.append({
            'phi': phi,
            'initial_cells': initial_cells,
            'final_cells': final_cells, 
            'delta': sim.current_delta,
            'stability': stability
        })
        
        print(f"φ={phi:.1f}: {initial_cells}→{final_cells} cells, Δ={sim.current_delta:.4f}, {stability}")
    
    print("\nConclusion: Higher φ values tend to maintain more dynamic patterns")
    print()

def example_6_pattern_analysis():
    """Example 6: Analyze specific patterns and their evolution"""
    print("Example 6: Pattern Analysis")  
    print("-" * 40)
    
    # Create simulator for pattern analysis
    sim = LifeFlux2DSimulator(width=20, height=20, use_life2=False, seed=42)
    
    # Set up a glider pattern
    glider_pattern = np.zeros((20, 20), dtype=np.int32)
    # Classic glider at position (5,5)
    glider_pattern[5, 7] = 1
    glider_pattern[6, 8] = 1  
    glider_pattern[7, 6:9] = 1
    
    if sim.use_cuda:
        import cupy as cp
        sim.grid = cp.asarray(glider_pattern)
    else:
        sim.grid = glider_pattern
    
    print("Analyzing glider pattern evolution:")
    print("Step | Live Cells | Center of Mass")
    print("-" * 35)
    
    for step in range(15):
        sim.step()
        
        grid_cpu = sim.to_cpu(sim.grid)
        live_cells = np.sum(grid_cpu)
        
        # Calculate center of mass
        y_coords, x_coords = np.where(grid_cpu == 1)
        if len(x_coords) > 0:
            cm_x = np.mean(x_coords)
            cm_y = np.mean(y_coords)
            print(f"{step:4d} | {live_cells:10d} | ({cm_x:4.1f}, {cm_y:4.1f})")
    
    print("\nThe glider should move diagonally with a period of 4 steps")
    print()

def main():
    """Run all examples"""
    print("LifeFlow Basic Usage Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_1_basic_conway()
    example_2_life2_with_flow()
    example_3_cuda_performance()
    example_4_visualization_modes()
    example_5_parameter_exploration()
    example_6_pattern_analysis()
    
    print("All examples completed successfully!")
    print("\nNext steps:")
    print("- Try the interactive mode: python src/life2.py --realtime")
    print("- Experiment with different parameters")  
    print("- Create your own custom patterns")
    print("- Explore 3D mode: python src/life3d.py")

if __name__ == "__main__":
    main()
