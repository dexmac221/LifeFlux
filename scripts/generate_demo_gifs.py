#!/usr/bin/env python3
"""
Generate demo GIFs for the LifeFlux README
Creates various demonstration animations showing different features
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lifeflux2d import LifeFlux2DSimulator
import numpy as np

def create_conway_demo():
    """Create a pure Conway's Game of Life demo"""
    print("Creating Conway's Life demo...")
    
    sim = LifeFlux2DSimulator(
        width=80, height=80,
        use_life2=False,  # Pure Conway
        phi_influence=0.0,
        seed=42,
        out_dir="assets",
        display_scale=2
    )
    
    # Start recording
    sim.running = True
    
    # Run for enough steps to show interesting patterns
    for _ in range(120):
        sim.step()
        
        # Stop early if converged
        if sim.is_converged:
            print(f"Conway demo converged at step {sim.step_count}")
            break
    
    sim.save_gifs()
    
    # Rename for clarity
    os.rename("assets/lifeflux2d_cells.gif", "assets/conway_life_demo.gif")
    print("✓ Conway's Life demo created")

def create_life2_flow_demo():
    """Create a Life 2.0 demo showing flow dynamics"""
    print("Creating Life 2.0 flow demo...")
    
    sim = LifeFlux2DSimulator(
        width=100, height=100,
        use_life2=True,
        phi_influence=0.4,  # Moderate field influence
        flow_gain=10.0,     # Strong flow visualization
        kernel_sigma=4.0,   # Smooth fields
        colored_dye=True,
        seed=123,
        out_dir="assets",
        display_scale=2
    )
    
    sim.running = True
    
    # Run until we get interesting flow patterns
    for step in range(150):
        sim.step()
        
        # Check for interesting dynamics (high flow speeds)
        if step > 20:
            flow_speed = np.sqrt(sim.to_cpu(sim.vx)**2 + sim.to_cpu(sim.vy)**2)
            max_flow = np.max(flow_speed)
            if max_flow < 0.1:  # If flow is too weak, continue
                continue
    
    sim.save_gifs()
    
    # The side-by-side GIF is already created by save_gifs
    print("✓ Life 2.0 flow demo created")

def create_vortex_demo():
    """Create a demo focusing on vortex formation"""
    print("Creating vortex formation demo...")
    
    sim = LifeFlux2DSimulator(
        width=120, height=120,
        use_life2=True,
        phi_influence=0.6,   # High field influence for chaos
        flow_gain=15.0,      # Very strong flow
        dt_flow=0.8,         # More aggressive advection
        kernel_sigma=5.0,    # Larger smoothing for vortices
        colored_dye=True,
        seed=456,
        out_dir="assets",
        display_scale=2
    )
    
    sim.running = True
    
    # Run longer to develop vortices
    for _ in range(200):
        sim.step()
    
    sim.save_gifs()
    
    # Rename the side-by-side for vortex demo
    if os.path.exists("assets/lifeflux2d_side_by_side.gif"):
        os.rename("assets/lifeflux2d_side_by_side.gif", "assets/vortex_formation_demo.gif")
    
    print("✓ Vortex formation demo created")

def create_comparison_demo():
    """Create a comparison between Conway and Life 2.0"""
    print("Creating Conway vs Life 2.0 comparison...")
    
    # First half: Pure Conway
    sim_conway = LifeFlux2DSimulator(
        width=60, height=60,
        use_life2=False,
        phi_influence=0.0,
        seed=789,
        out_dir="assets",
        display_scale=3
    )
    
    sim_conway.running = True
    conway_frames = []
    
    for _ in range(60):
        cell_rgb, _, _ = sim_conway.step()
        conway_frames.append(cell_rgb)
    
    # Second half: Life 2.0
    sim_life2 = LifeFlux2DSimulator(
        width=60, height=60,
        use_life2=True,
        phi_influence=0.5,
        flow_gain=12.0,
        colored_dye=True,
        seed=789,  # Same seed for comparison
        out_dir="assets",
        display_scale=3
    )
    
    sim_life2.running = True
    life2_cells = []
    life2_flow = []
    
    for _ in range(60):
        cell_rgb, dye_rgb, _ = sim_life2.step()
        life2_cells.append(cell_rgb)
        life2_flow.append(dye_rgb)
    
    # Create comparison frames
    import imageio.v3 as iio
    comparison_frames = []
    
    for i in range(60):
        # Create 2x2 grid: Conway top-left, Life2 cells top-right, Life2 flow bottom spanning both
        conway_frame = conway_frames[i]
        life2_cell_frame = life2_cells[i]
        life2_flow_frame = life2_flow[i]
        
        # Ensure same size
        h, w = conway_frame.shape[:2]
        
        # Create top row: Conway | Life2 cells
        top_row = np.concatenate([conway_frame, life2_cell_frame], axis=1)
        
        # Create bottom row: Life2 flow (stretched to full width)
        flow_stretched = np.repeat(life2_flow_frame, 2, axis=1)[:, :top_row.shape[1], :]
        
        # Combine with separator
        separator = np.ones((4, top_row.shape[1], 3), dtype=np.uint8) * 64
        comparison_frame = np.concatenate([top_row, separator, flow_stretched], axis=0)
        comparison_frames.append(comparison_frame)
    
    iio.imwrite("assets/conway_vs_life2_comparison.gif", comparison_frames, duration=0.08, loop=0)
    print("✓ Comparison demo created")

def create_parameter_sweep_demo():
    """Create a demo showing different phi_influence values"""
    print("Creating parameter sweep demo...")
    
    phi_values = [0.0, 0.2, 0.5, 0.8]
    all_frames = []
    
    for phi in phi_values:
        print(f"  Testing phi_influence = {phi}")
        
        sim = LifeFlux2DSimulator(
            width=50, height=50,
            use_life2=True,
            phi_influence=phi,
            flow_gain=8.0,
            colored_dye=True,
            seed=999,  # Same seed for all
            out_dir="assets",
            display_scale=2
        )
        
        # Run for consistent number of steps
        frames_for_phi = []
        for step in range(40):
            cell_rgb, dye_rgb, _ = sim.step()
            
            # Create side-by-side for this phi value
            separator = np.ones((cell_rgb.shape[0], 2, 3), dtype=np.uint8) * 128
            combined = np.concatenate([cell_rgb, separator, dye_rgb], axis=1)
            frames_for_phi.append(combined)
        
        all_frames.extend(frames_for_phi)
    
    import imageio.v3 as iio
    iio.imwrite("assets/parameter_sweep_demo.gif", all_frames, duration=0.06, loop=0)
    print("✓ Parameter sweep demo created")

def main():
    """Generate all demo GIFs"""
    print("Generating LifeFlux Demo GIFs")
    print("=" * 40)
    
    # Create assets directory
    os.makedirs("assets", exist_ok=True)
    
    try:
        # Generate different types of demos
        create_conway_demo()
        create_life2_flow_demo() 
        create_vortex_demo()
        create_comparison_demo()
        create_parameter_sweep_demo()
        
        print("\n" + "=" * 40)
        print("✅ All demo GIFs generated successfully!")
        print("\nGenerated files in assets/:")
        
        # List generated files
        for filename in sorted(os.listdir("assets")):
            if filename.endswith(".gif"):
                filepath = os.path.join("assets", filename)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  {filename}: {size_kb:.1f} KB")
        
        print(f"\n📁 Ready to add to README.md!")
        print("Use these GIFs to showcase LifeFlux features in your documentation.")
        
    except Exception as e:
        print(f"❌ Error generating demos: {e}")
        raise

if __name__ == "__main__":
    main()
