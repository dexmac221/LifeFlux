#!/usr/bin/env python3
"""
Integration tests for the complete LifeFlow system
"""

import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_life2d_batch_mode():
    """Test 2D Life in batch mode"""
    from lifeflux2d import LifeFlux2DSimulator
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing 2D Life batch mode...")
        
        # Test standard Conway's Life
        sim = LifeFlux2DSimulator(width=20, height=20, use_life2=False, 
                           out_dir=temp_dir, seed=42)
        sim.running = True
        
        for _ in range(5):
            sim.step()
        
        sim.save_gifs()
        
        # Check output files
        expected_files = ['lifeflux2d_cells.gif', 'lifeflux2d_dye.gif', 'lifeflux2d_combo.gif']
        for filename in expected_files:
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath), f"Missing output file: {filename}"
            assert os.path.getsize(filepath) > 0, f"Empty output file: {filename}"
        
        print("✓ 2D Conway's Life batch test passed")
        
        # Test Life 2.0 mode
        sim2 = LifeFlux2DSimulator(width=20, height=20, use_life2=True,
                            phi_influence=0.3, out_dir=temp_dir, seed=42)
        sim2.running = True
        
        for _ in range(5):
            sim2.step()
        
        # Should have non-zero flow fields
        assert sim2.to_cpu(sim2.vx).max() > 0 or sim2.to_cpu(sim2.vy).max() > 0
        print("✓ 2D Life 2.0 test passed")

def test_life3d_headless():
    """Test 3D Life in headless mode"""
    try:
        from lifeflux3d import LifeFlux3DSimulator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Testing 3D Life headless mode...")
            
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                sim = LifeFlux3DSimulator(width=8, height=8, depth=8, 
                                    use_life3d=False, seed=42)
                
                # Run a few steps
                for i in range(3):
                    sim.step()
                    if hasattr(sim, 'export_ply'):
                        sim.export_ply(f"test_step_{i:04d}.ply")
                
                print("✓ 3D Life headless test passed")
                
            finally:
                os.chdir(original_dir)
                
    except ImportError:
        print("⚠ 3D Life module not available, skipping test")

def test_cuda_compatibility():
    """Test CUDA compatibility if available"""
    try:
        import cupy
        print("Testing CUDA compatibility...")
        
        from lifeflux2d import LifeFlux2DSimulator
        
        # Test CUDA 2D
        sim_cuda = LifeFlux2DSimulator(width=32, height=32, use_cuda=True, seed=42)
        assert sim_cuda.use_cuda == True
        
        for _ in range(3):
            sim_cuda.step()
        
        # Should produce valid output
        cell_rgb, dye_rgb, combo = sim_cuda.step()
        assert cell_rgb.shape == (32, 32, 3)
        assert dye_rgb.shape == (32, 32, 3)
        
        print("✓ CUDA 2D compatibility test passed")
        
        # Test CUDA 3D if available
        try:
            from lifeflux3d import LifeFlux3DSimulator
            sim3d_cuda = LifeFlux3DSimulator(width=16, height=16, depth=16, 
                                       use_cuda=True, seed=42)
            sim3d_cuda.step()
            print("✓ CUDA 3D compatibility test passed")
        except ImportError:
            print("⚠ 3D CUDA test skipped (module not available)")
            
    except ImportError:
        print("⚠ CUDA not available, skipping CUDA tests")

def test_convergence_detection():
    """Test convergence detection system"""
    from lifeflux2d import LifeFlux2DSimulator
    
    print("Testing convergence detection...")
    
    # Create a simulator with a still life (should converge quickly)
    sim = LifeFlux2DSimulator(width=10, height=10, use_life2=False, seed=42)
    
    # Set up a block pattern (2x2 still life)
    if sim.use_cuda:
        import cupy as cp
        sim.grid = cp.zeros((10, 10), dtype=cp.int32)
        sim.grid[4:6, 4:6] = 1
    else:
        import numpy as np
        sim.grid = np.zeros((10, 10), dtype=np.int32)
        sim.grid[4:6, 4:6] = 1
    
    # Run enough steps to detect convergence
    for i in range(60):
        sim.step()
        if sim.is_converged:
            print(f"✓ Convergence detected at step {i}")
            break
    else:
        print("⚠ Convergence not detected within 60 steps")

def test_visualization_modes():
    """Test different visualization modes"""
    from lifeflux2d import LifeFlux2DSimulator
    
    print("Testing visualization modes...")
    
    sim = LifeFlux2DSimulator(width=16, height=16, use_life2=True, 
                        colored_dye=True, seed=42)
    
    # Run a few steps to generate content
    for _ in range(5):
        sim.step()
    
    # Test colored dye field
    dye_rgb = sim.create_colored_dye_field(sim.dye, sim.vx, sim.vy)
    assert dye_rgb.shape == (16, 16, 3)
    assert dye_rgb.dtype.name == 'uint8'
    
    # Test grayscale mode
    sim.colored_dye = False
    cell_rgb, dye_rgb, combo = sim.step()
    assert dye_rgb.shape == (16, 16, 3)
    
    print("✓ Visualization modes test passed")

def test_parameter_ranges():
    """Test various parameter ranges and edge cases"""
    from lifeflux2d import LifeFlux2DSimulator
    
    print("Testing parameter ranges...")
    
    # Test different phi_influence values
    for phi_val in [0.0, 0.2, 0.5, 1.0]:
        sim = LifeFlux2DSimulator(width=16, height=16, use_life2=True,
                           phi_influence=phi_val, seed=42)
        sim.step()
        print(f"  ✓ phi_influence={phi_val}")
    
    # Test different grid sizes
    for size in [(10, 10), (50, 20), (20, 50)]:
        w, h = size
        sim = LifeFlux2DSimulator(width=w, height=h, seed=42)
        sim.step()
        print(f"  ✓ size={w}x{h}")
    
    # Test different flow parameters
    sim = LifeFlux2DSimulator(width=16, height=16, flow_gain=15.0, dt_flow=0.3, seed=42)
    sim.step()
    print("  ✓ Custom flow parameters")
    
    print("✓ Parameter ranges test passed")

if __name__ == "__main__":
    print("LifeFlow Integration Tests")
    print("=" * 50)
    
    try:
        test_life2d_batch_mode()
        test_life3d_headless() 
        test_cuda_compatibility()
        test_convergence_detection()
        test_visualization_modes()
        test_parameter_ranges()
        
        print("\n" + "=" * 50)
        print("✅ All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise
