#!/usr/bin/env python3
"""
Test suite for Life2D simulator
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lifeflux2d import LifeFlux2DSimulator

class TestLifeFlux2DSimulator:
    """Test cases for the LifeFlux2DSimulator class"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        sim = LifeFlux2DSimulator(width=50, height=50, seed=42)
        assert sim.width == 50
        assert sim.height == 50
        assert sim.grid.shape == (50, 50)
        assert sim.dye.shape == (50, 50)
        assert sim.phi.shape == (50, 50)
    
    def test_cuda_initialization(self):
        """Test CUDA initialization if available"""
        try:
            import cupy
            sim = LifeFlux2DSimulator(width=32, height=32, use_cuda=True, seed=42)
            assert sim.use_cuda == True
            assert hasattr(sim.grid, 'device')  # CuPy arrays have device attribute
        except ImportError:
            # Skip if CUDA not available
            pytest.skip("CUDA not available")
    
    def test_neighbor_counting(self):
        """Test neighbor counting algorithm"""
        sim = LifeFlux2DSimulator(width=5, height=5, seed=42)
        
        # Create a simple pattern (glider)
        if sim.use_cuda:
            import cupy as cp
            sim.grid = cp.zeros((5, 5), dtype=cp.int32)
            sim.grid[1, 2] = 1
            sim.grid[2, 3] = 1
            sim.grid[3, 1:4] = 1
        else:
            sim.grid = np.zeros((5, 5), dtype=np.int32)
            sim.grid[1, 2] = 1
            sim.grid[2, 3] = 1
            sim.grid[3, 1:4] = 1
        
        neighbors = sim.neighbor_sum(sim.grid)
        
        # Convert to CPU for testing if needed
        if sim.use_cuda:
            import cupy as cp
            neighbors = cp.asnumpy(neighbors)
            grid_cpu = cp.asnumpy(sim.grid)
        else:
            grid_cpu = sim.grid
        
        # Test specific neighbor counts for glider pattern
        assert neighbors[2, 2] == 5  # Center cell should have 5 neighbors
        assert neighbors[0, 0] == 0  # Corner should have 0 neighbors
    
    def test_conway_rules(self):
        """Test standard Conway's Game of Life rules"""
        sim = LifeFlux2DSimulator(width=10, height=10, use_life2=False, seed=42)
        
        initial_state = sim.to_cpu(sim.grid.copy())
        sim.step()
        new_state = sim.to_cpu(sim.grid)
        
        # Grid should change after one step (with high probability)
        assert not np.array_equal(initial_state, new_state)
    
    def test_life2_rules(self):
        """Test Life 2.0 rules with potential field"""
        sim = LifeFlux2DSimulator(width=10, height=10, use_life2=True, 
                           phi_influence=0.5, seed=42)
        
        initial_state = sim.to_cpu(sim.grid.copy())
        sim.step()
        new_state = sim.to_cpu(sim.grid)
        
        # Should have non-zero potential field
        phi_cpu = sim.to_cpu(sim.phi)
        assert np.any(phi_cpu > 0)
        
        # Grid should change
        assert not np.array_equal(initial_state, new_state)
    
    def test_dye_advection(self):
        """Test dye field advection"""
        sim = LifeFlux2DSimulator(width=20, height=20, use_life2=True, seed=42)
        
        initial_dye = sim.to_cpu(sim.dye.copy())
        sim.step()
        new_dye = sim.to_cpu(sim.dye)
        
        # Dye field should change due to advection
        assert not np.array_equal(initial_dye, new_dye)
    
    def test_convergence_detection(self):
        """Test convergence detection mechanism"""
        # Create a still life pattern (block)
        sim = LifeFlux2DSimulator(width=10, height=10, use_life2=False, seed=42)
        
        if sim.use_cuda:
            import cupy as cp
            sim.grid = cp.zeros((10, 10), dtype=cp.int32)
            sim.grid[4:6, 4:6] = 1  # 2x2 block (still life)
        else:
            sim.grid = np.zeros((10, 10), dtype=np.int32)
            sim.grid[4:6, 4:6] = 1  # 2x2 block (still life)
        
        # Run enough steps to detect convergence
        for _ in range(55):
            sim.step()
        
        # Should detect convergence for still life
        assert sim.is_converged
    
    def test_gif_generation(self):
        """Test GIF output generation"""
        sim = LifeFlux2DSimulator(width=10, height=10, seed=42, out_dir="test_output")
        sim.running = True
        
        # Generate some frames
        for _ in range(5):
            sim.step()
        
        assert len(sim.cells_frames) == 5
        assert len(sim.dye_frames) == 5
        assert len(sim.combo_frames) == 5
    
    def test_colored_dye_field(self):
        """Test colored dye field generation"""
        sim = LifeFlux2DSimulator(width=10, height=10, use_life2=True, 
                           colored_dye=True, seed=42)
        
        # Create some flow
        for _ in range(3):
            sim.step()
        
        # Generate colored dye field
        dye_rgb = sim.create_colored_dye_field(sim.dye, sim.vx, sim.vy)
        
        assert dye_rgb.shape == (10, 10, 3)
        assert dye_rgb.dtype == np.uint8
        assert np.max(dye_rgb) <= 255
        assert np.min(dye_rgb) >= 0
    
    def test_scale_image(self):
        """Test image scaling functionality"""
        sim = LifeFlux2DSimulator(width=10, height=10, display_scale=2, seed=42)
        
        # Create test image
        test_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        scaled = sim.scale_image(test_img)
        
        assert scaled.shape == (20, 20, 3)
        
        # Test grayscale scaling
        gray_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        scaled_gray = sim.scale_image(gray_img)
        assert scaled_gray.shape == (20, 20)

    def test_reset_functionality(self):
        """Test simulator reset"""
        sim = LifeFlux2DSimulator(width=10, height=10, seed=42)
        
        # Run some steps
        for _ in range(5):
            sim.step()
        
        initial_step = sim.step_count
        sim.reset()
        
        assert sim.step_count == 0
        assert len(sim.cells_frames) == 0
        assert len(sim.convergence_window) == 0

if __name__ == "__main__":
    pytest.main([__file__])
