#!/usr/bin/env python3
"""
Test suite for Life3D simulator
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from lifeflux3d import LifeFlux3DSimulator
    LIFE3D_AVAILABLE = True
except ImportError:
    LIFE3D_AVAILABLE = False

@pytest.mark.skipif(not LIFE3D_AVAILABLE, reason="lifeflow3d module not available")
class TestLifeFlux3DSimulator:
    """Test cases for the LifeFlux3DSimulator class"""
    
    def test_initialization(self):
        """Test 3D simulator initialization"""
        sim = LifeFlux3DSimulator(width=16, height=16, depth=16, seed=42)
        assert sim.width == 16
        assert sim.height == 16
        assert sim.depth == 16
        assert sim.cells.shape == (16, 16, 16)
        
    def test_cuda_initialization_3d(self):
        """Test CUDA initialization for 3D if available"""
        try:
            import cupy
            sim = LifeFlux3DSimulator(width=8, height=8, depth=8, use_cuda=True, seed=42)
            assert sim.use_cuda == True
        except ImportError:
            pytest.skip("CUDA not available")
    
    def test_3d_neighbor_counting(self):
        """Test 3D neighbor counting (26-connectivity)"""
        sim = LifeFlux3DSimulator(width=5, height=5, depth=5, seed=42)
        
        # Create a simple 3D pattern
        if sim.use_cuda:
            import cupy as cp
            sim.cells = cp.zeros((5, 5, 5), dtype=bool)
            sim.cells[2, 2, 2] = True  # Single cell in center
        else:
            sim.cells = np.zeros((5, 5, 5), dtype=bool)
            sim.cells[2, 2, 2] = True  # Single cell in center
        
        neighbors = sim.count_3d_neighbors(sim.cells)
        
        if sim.use_cuda:
            import cupy as cp
            neighbors = cp.asnumpy(neighbors)
        
        # Center cell should have 0 neighbors, surrounding cells should have 1
        assert neighbors[2, 2, 2] == 0
        assert neighbors[1, 2, 2] == 1  # Adjacent cell
        assert neighbors[1, 1, 1] == 1  # Diagonal cell
    
    def test_conway3d_rules(self):
        """Test 3D Conway rules"""
        sim = LifeFlux3DSimulator(width=10, height=10, depth=10, 
                            use_life3d=False, seed=42)
        
        # Convert to CPU for counting
        if sim.use_cuda:
            import cupy as cp
            initial_count = int(cp.sum(sim.cells).get())
        else:
            initial_count = int(np.sum(sim.cells))
        
        sim.step()
        
        if sim.use_cuda:
            new_count = int(cp.sum(sim.cells).get())
        else:
            new_count = int(np.sum(sim.cells))
        
        # Cell count should change
        assert initial_count != new_count
    
    def test_life3d_rules(self):
        """Test 3D Life 2.0 rules with potential field"""
        sim = LifeFlux3DSimulator(width=8, height=8, depth=8, 
                            use_life3d=True, phi_influence=0.5, seed=42)
        
        sim.step()
        
        # Should have non-zero potential field
        if sim.use_cuda:
            import cupy as cp
            phi_cpu = cp.asnumpy(sim.phi)
        else:
            phi_cpu = sim.phi
        
        assert np.any(phi_cpu > 0)
    
    def test_3d_potential_field(self):
        """Test 3D potential field computation"""
        sim = LifeFlux3DSimulator(width=8, height=8, depth=8, use_life3d=True, seed=42)
        
        # Run one step to compute potential field
        sim.step()
        
        if sim.use_cuda:
            import cupy as cp
            phi_cpu = cp.asnumpy(sim.phi)
            cells_cpu = cp.asnumpy(sim.cells)
        else:
            phi_cpu = sim.phi
            cells_cpu = sim.cells
        
        # Phi should be non-negative
        assert np.all(phi_cpu >= 0)
        
        # Should have some non-zero values where cells exist
        if np.any(cells_cpu):
            assert np.any(phi_cpu > 0)
    
    def test_convergence_3d(self):
        """Test 3D convergence detection"""
        sim = LifeFlux3DSimulator(width=6, height=6, depth=6, use_life3d=False, seed=42)
        
        # Create a stable 3D pattern (if possible)
        if sim.use_cuda:
            import cupy as cp
            sim.cells = cp.zeros((6, 6, 6), dtype=bool)
        else:
            sim.cells = np.zeros((6, 6, 6), dtype=bool)
        
        # Run steps to test convergence detection
        for _ in range(10):
            sim.step()
        
        # Should track convergence
        assert hasattr(sim, 'is_converged')
        assert hasattr(sim, 'current_delta')

if __name__ == "__main__":
    pytest.main([__file__])
