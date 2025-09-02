#!/usr/bin/env python3
"""
Generate a perfect demo GIF for the README showing side-by-side Life and Flow
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lifeflux2d import LifeFlux2DSimulator
import numpy as np
import imageio.v3 as iio

def create_readme_demo():
    """Create the perfect demo for README"""
    print("Creating README demo GIF...")
    
    # Use parameters that create nice visuals
    sim = LifeFlux2DSimulator(
        width=120, height=120,
        use_life2=True,
        phi_influence=0.5,   # Balanced field influence
        flow_gain=12.0,      # Strong but not chaotic flow
        kernel_sigma=4.5,    # Good smoothing for vortices
        dt_flow=0.75,        # Moderate advection
        colored_dye=True,
        seed=1337,           # Seed that produces interesting patterns
        out_dir="assets",
        display_scale=1      # Keep original resolution
    )
    
    print("Recording simulation...")
    sim.running = True
    
    # Record frames
    frames_to_record = 100
    for step in range(frames_to_record):
        sim.step()
        
        if (step + 1) % 20 == 0:
            print(f"  Progress: {step + 1}/{frames_to_record} frames")
    
    # Create enhanced side-by-side with labels
    print("Creating enhanced side-by-side demo...")
    
    demo_frames = []
    for i, (cell_frame, dye_frame) in enumerate(zip(sim.cells_frames, sim.dye_frames)):
        # Ensure both frames are RGB
        if len(cell_frame.shape) == 2:
            cell_frame = np.stack([cell_frame] * 3, axis=-1)
        if len(dye_frame.shape) == 2:
            dye_frame = np.stack([dye_frame] * 3, axis=-1)
        
        # Create separator with gradient
        h = cell_frame.shape[0]
        separator_width = 8
        separator = np.zeros((h, separator_width, 3), dtype=np.uint8)
        # Add a subtle gradient separator
        for y in range(h):
            intensity = int(128 + 64 * np.sin(2 * np.pi * y / h))
            separator[y, :] = [intensity//3, intensity//2, intensity]
        
        # Combine frames
        combined = np.concatenate([cell_frame, separator, dye_frame], axis=1)
        
        # Add text overlay (simple approach)
        combined_with_text = add_simple_text_overlay(combined, 
                                                   f"LifeFlux Demo - Step {i+1}", 
                                                   "Cellular Life", "Flow Dynamics")
        
        demo_frames.append(combined_with_text)
    
    # Save the main demo
    demo_path = "assets/lifeflux_readme_demo.gif"
    iio.imwrite(demo_path, demo_frames, duration=0.1, loop=0)
    
    # Also save a faster version
    fast_demo_frames = demo_frames[::2]  # Every other frame
    fast_demo_path = "assets/lifeflux_readme_demo_fast.gif"
    iio.imwrite(fast_demo_path, fast_demo_frames, duration=0.08, loop=0)
    
    file_size = os.path.getsize(demo_path) / 1024
    fast_size = os.path.getsize(fast_demo_path) / 1024
    
    print(f"✅ Demo GIFs created:")
    print(f"  Main demo: {demo_path} ({file_size:.1f} KB)")
    print(f"  Fast demo: {fast_demo_path} ({fast_size:.1f} KB)")
    
    return demo_path, fast_demo_path

def add_simple_text_overlay(image, title, left_label, right_label):
    """Add simple text overlay using basic image manipulation"""
    h, w, c = image.shape
    result = image.copy()
    
    # Add a dark banner at the top for title
    banner_height = 25
    result[:banner_height, :] = [20, 30, 50]  # Dark blue banner
    
    # Add title (simple approach - just brighten some pixels to form letters)
    title_y = 12
    title_start_x = w // 2 - len(title) * 3
    
    # Simple pixel text (just make bright spots)
    for i, char in enumerate(title[:min(len(title), w//6)]):
        x = title_start_x + i * 6
        if 0 <= x < w - 5 and char != ' ':
            # Make a simple 3x5 bright rectangle for each character
            result[title_y-2:title_y+3, x:x+3] = [200, 220, 255]
    
    # Add labels for left and right sides
    cell_center = w // 4
    dye_center = 3 * w // 4
    
    label_y = banner_height + 15
    
    # Left label
    for i, char in enumerate(left_label[:min(len(left_label), w//8)]):
        x = cell_center - len(left_label) * 2 + i * 4
        if 0 <= x < w - 3 and char != ' ':
            result[label_y-1:label_y+2, x:x+2] = [255, 200, 100]  # Orange
    
    # Right label  
    for i, char in enumerate(right_label[:min(len(right_label), w//8)]):
        x = dye_center - len(right_label) * 2 + i * 4
        if 0 <= x < w - 3 and char != ' ':
            result[label_y-1:label_y+2, x:x+2] = [100, 255, 200]  # Cyan
    
    return result

def create_showcase_variations():
    """Create variations for different showcase needs"""
    print("\nCreating showcase variations...")
    
    # Small version for compact display
    sim_small = LifeFlux2DSimulator(
        width=60, height=60,
        use_life2=True,
        phi_influence=0.4,
        flow_gain=10.0,
        colored_dye=True,
        seed=2023,
        out_dir="assets",
        display_scale=2
    )
    
    sim_small.running = True
    small_frames = []
    
    for _ in range(50):
        cell_rgb, dye_rgb, _ = sim_small.step()
        
        # Simple side-by-side
        separator = np.ones((cell_rgb.shape[0], 3, 3), dtype=np.uint8) * 100
        combined = np.concatenate([cell_rgb, separator, dye_rgb], axis=1)
        small_frames.append(combined)
    
    iio.imwrite("assets/lifeflux_compact_demo.gif", small_frames, duration=0.12, loop=0)
    
    # High-res version for detailed viewing
    sim_hires = LifeFlux2DSimulator(
        width=80, height=80,
        use_life2=True,
        phi_influence=0.6,
        flow_gain=15.0,
        kernel_sigma=3.0,
        colored_dye=True,
        seed=4567,
        out_dir="assets",
        display_scale=3
    )
    
    sim_hires.running = True
    hires_frames = []
    
    for _ in range(75):
        cell_rgb, dye_rgb, _ = sim_hires.step()
        
        # Create high-quality side-by-side
        separator = np.zeros((cell_rgb.shape[0], 6, 3), dtype=np.uint8)
        separator[:, 2:4] = [80, 120, 160]  # Blue separator
        
        combined = np.concatenate([cell_rgb, separator, dye_rgb], axis=1)
        hires_frames.append(combined)
    
    iio.imwrite("assets/lifeflux_hires_demo.gif", hires_frames, duration=0.08, loop=0)
    
    print("✅ Showcase variations created")

def main():
    print("Creating LifeFlux README Demo")
    print("=" * 40)
    
    os.makedirs("assets", exist_ok=True)
    
    # Create the main README demo
    main_demo, fast_demo = create_readme_demo()
    
    # Create variations
    create_showcase_variations()
    
    print("\n" + "=" * 40)
    print("📱 Ready for README!")
    print("\nSuggested usage in README.md:")
    print(f"![LifeFlux Demo]({main_demo})")
    print("\nOr for a more compact version:")
    print("![LifeFlux Demo](assets/lifeflux_compact_demo.gif)")
    
    print(f"\n📁 All files in assets/:")
    for filename in sorted(os.listdir("assets")):
        if filename.endswith(".gif"):
            filepath = os.path.join("assets", filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {filename}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
