#!/usr/bin/env python3
"""
Life3D - 3D Cellular Automaton with Volumetric Dye Field Visualization
Extended from Life 2.0 into full 3D space with OpenGL rendering.

Features:
- 3D Conway's Life with 26-neighbor rules
- 3D Life 2.0 with potential field and volumetric dye advection  
- CUDA acceleration for large 3D grids
- Real-time Open3D visualization with volumetric rendering
- Interactive 3D controls and camera manipulation
- Configurable 3D rules and parameters
"""

import numpy as np
import open3d as o3d
import argparse
import time
import threading
from typing import Tuple, Optional
import math

# CUDA imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    CUDA_AVAILABLE = True
    print("CUDA acceleration available!")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available, using CPU")


class LifeFlux3DSimulator:
    """3D Cellular Automaton with volumetric dye field simulation"""
    
    def __init__(self, width=64, height=64, depth=64, use_life3d=True, 
                 phi_influence=0.5, kernel_sigma=2.0, use_cuda=False, seed=None):
        self.width = width
        self.height = height  
        self.depth = depth
        self.use_life3d = use_life3d
        self.phi_influence = phi_influence
        self.kernel_sigma = kernel_sigma
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        if seed is not None:
            np.random.seed(seed)
            if self.use_cuda:
                cp.random.seed(seed)
        
        # Initialize arrays
        if self.use_cuda:
            self.cells = cp.random.random((depth, height, width)) > 0.7
            self.phi = cp.zeros((depth, height, width), dtype=cp.float32)
            self.dye = cp.zeros((depth, height, width, 3), dtype=cp.float32)
            if use_life3d:
                self.dye_velocity = cp.zeros((depth, height, width, 3), dtype=cp.float32)
        else:
            self.cells = np.random.random((depth, height, width)) > 0.7
            self.phi = np.zeros((depth, height, width), dtype=np.float32)
            self.dye = np.zeros((depth, height, width, 3), dtype=np.float32)
            if use_life3d:
                self.dye_velocity = np.zeros((depth, height, width, 3), dtype=np.float32)
        
        self.step_count = 0
        self.is_converged = False
        self.current_delta = 0.0
        self.convergence_history = []
        
        # 3D Gaussian kernel for potential field
        self._create_3d_kernel()
    
    def _create_3d_kernel(self):
        """Create 3D Gaussian kernel for potential field computation"""
        size = max(5, int(self.kernel_sigma * 4))
        if size % 2 == 0:
            size += 1
        
        center = size // 2
        if self.use_cuda:
            z, y, x = cp.mgrid[0:size, 0:size, 0:size]
            kernel = cp.exp(-((x-center)**2 + (y-center)**2 + (z-center)**2) / (2 * self.kernel_sigma**2))
            self.kernel = kernel / cp.sum(kernel)
        else:
            z, y, x = np.mgrid[0:size, 0:size, 0:size]
            kernel = np.exp(-((x-center)**2 + (y-center)**2 + (z-center)**2) / (2 * self.kernel_sigma**2))
            self.kernel = kernel / np.sum(kernel)
    
    def count_3d_neighbors(self, cells):
        """Count 26-connected neighbors in 3D space"""
        if self.use_cuda:
            neighbors = cp.zeros_like(cells, dtype=cp.int32)
            # Use 3x3x3 convolution kernel excluding center
            kernel = cp.ones((3, 3, 3), dtype=cp.int32)
            kernel[1, 1, 1] = 0  # Exclude center cell
            neighbors = cupy_ndimage.convolve(cells.astype(cp.int32), kernel, mode='wrap')
        else:
            from scipy import ndimage
            neighbors = np.zeros_like(cells, dtype=np.int32)
            kernel = np.ones((3, 3, 3), dtype=np.int32)
            kernel[1, 1, 1] = 0  # Exclude center cell
            neighbors = ndimage.convolve(cells.astype(np.int32), kernel, mode='wrap')
        
        return neighbors
    
    def apply_3d_life_rules(self, cells, neighbors):
        """Apply 3D Conway's Life rules (adapted for 3D: survive with 4-7 neighbors, born with 6-7)"""
        if self.use_cuda:
            new_cells = cp.zeros_like(cells, dtype=bool)
            # Birth: dead cell with 6-7 neighbors becomes alive
            new_cells |= (~cells) & ((neighbors >= 6) & (neighbors <= 7))
            # Survival: live cell with 4-7 neighbors survives
            new_cells |= cells & ((neighbors >= 4) & (neighbors <= 7))
        else:
            new_cells = np.zeros_like(cells, dtype=bool)
            # Birth: dead cell with 6-7 neighbors becomes alive  
            new_cells |= (~cells) & ((neighbors >= 6) & (neighbors <= 7))
            # Survival: live cell with 4-7 neighbors survives
            new_cells |= cells & ((neighbors >= 4) & (neighbors <= 7))
        
        return new_cells
    
    def apply_3d_life2_rules(self, cells, neighbors, phi):
        """Apply 3D Life 2.0 rules with potential field influence"""
        if self.use_cuda:
            # Base 3D Conway rules
            conway_cells = self.apply_3d_life_rules(cells, neighbors)
            
            # Phi influence: high potential encourages birth, low potential death
            phi_normalized = (phi - cp.min(phi)) / (cp.max(phi) - cp.min(phi) + 1e-8)
            birth_prob = 0.1 + 0.4 * phi_normalized  # 0.1 to 0.5
            death_prob = 0.05 + 0.15 * (1 - phi_normalized)  # 0.05 to 0.2
            
            # Apply phi influence
            random_vals = cp.random.random(cells.shape)
            phi_birth = (~cells) & (random_vals < birth_prob * self.phi_influence)
            phi_death = cells & (random_vals < death_prob * self.phi_influence)
            
            # Combine Conway rules with phi influence
            new_cells = conway_cells | phi_birth
            new_cells = new_cells & (~phi_death)
        else:
            # Base 3D Conway rules
            conway_cells = self.apply_3d_life_rules(cells, neighbors)
            
            # Phi influence
            phi_normalized = (phi - np.min(phi)) / (np.max(phi) - np.min(phi) + 1e-8)
            birth_prob = 0.1 + 0.4 * phi_normalized
            death_prob = 0.05 + 0.15 * (1 - phi_normalized)
            
            # Apply phi influence
            random_vals = np.random.random(cells.shape)
            phi_birth = (~cells) & (random_vals < birth_prob * self.phi_influence)
            phi_death = cells & (random_vals < death_prob * self.phi_influence)
            
            # Combine rules
            new_cells = conway_cells | phi_birth
            new_cells = new_cells & (~phi_death)
        
        return new_cells
    
    def blur_3d(self, field):
        """Apply 3D Gaussian blur to field"""
        if self.use_cuda:
            return cupy_ndimage.convolve(field, self.kernel, mode='wrap')
        else:
            from scipy import ndimage
            return ndimage.convolve(field, self.kernel, mode='wrap')
    
    def compute_3d_gradient(self, field):
        """Compute 3D gradient for velocity field"""
        if self.use_cuda:
            gz, gy, gx = cp.gradient(field)
            return cp.stack([gx, gy, gz], axis=-1)
        else:
            gz, gy, gx = np.gradient(field)
            return np.stack([gx, gy, gz], axis=-1)
    
    def advect_3d_dye(self, dye, velocity, dt=0.1):
        """Advect 3D dye field using Semi-Lagrangian method"""
        if self.use_cuda:
            d, h, w = dye.shape[:3]
            z, y, x = cp.mgrid[0:d, 0:h, 0:w]
            
            # Backward trace
            back_x = x - velocity[:,:,:,0] * dt
            back_y = y - velocity[:,:,:,1] * dt  
            back_z = z - velocity[:,:,:,2] * dt
            
            # Wrap coordinates
            back_x = cp.mod(back_x, w)
            back_y = cp.mod(back_y, h)
            back_z = cp.mod(back_z, d)
            
            # Trilinear interpolation
            x0 = cp.floor(back_x).astype(cp.int32)
            x1 = cp.mod(x0 + 1, w)
            y0 = cp.floor(back_y).astype(cp.int32)  
            y1 = cp.mod(y0 + 1, h)
            z0 = cp.floor(back_z).astype(cp.int32)
            z1 = cp.mod(z0 + 1, d)
            
            xd = back_x - x0
            yd = back_y - y0
            zd = back_z - z0
            
            # Expand dimensions for broadcasting with color channels
            xd = xd[..., cp.newaxis]
            yd = yd[..., cp.newaxis]
            zd = zd[..., cp.newaxis]
            
            # Trilinear interpolation weights
            c000 = dye[z0, y0, x0] * (1-xd) * (1-yd) * (1-zd)
            c001 = dye[z1, y0, x0] * (1-xd) * (1-yd) * zd
            c010 = dye[z0, y1, x0] * (1-xd) * yd * (1-zd)
            c011 = dye[z1, y1, x0] * (1-xd) * yd * zd
            c100 = dye[z0, y0, x1] * xd * (1-yd) * (1-zd)
            c101 = dye[z1, y0, x1] * xd * (1-yd) * zd
            c110 = dye[z0, y1, x1] * xd * yd * (1-zd)
            c111 = dye[z1, y1, x1] * xd * yd * zd
            
            new_dye = c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111
        else:
            from scipy import ndimage
            d, h, w = dye.shape[:3]
            z, y, x = np.mgrid[0:d, 0:h, 0:w]
            
            # Backward trace
            back_x = x - velocity[:,:,:,0] * dt
            back_y = y - velocity[:,:,:,1] * dt
            back_z = z - velocity[:,:,:,2] * dt
            
            # Simple trilinear interpolation using scipy
            new_dye = np.zeros_like(dye)
            for c in range(dye.shape[3]):
                coords = np.array([back_z.flatten(), back_y.flatten(), back_x.flatten()])
                new_dye[:,:,:,c] = ndimage.map_coordinates(
                    dye[:,:,:,c], coords, order=1, mode='wrap'
                ).reshape(d, h, w)
        
        return new_dye
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one simulation step and return visualization data"""
        # Count neighbors
        neighbors = self.count_3d_neighbors(self.cells)
        
        if self.use_life3d:
            # Update potential field
            self.phi = self.blur_3d(self.cells.astype(np.float32 if not self.use_cuda else cp.float32))
            
            # Apply Life 2.0 rules
            old_cells = self.cells.copy()
            self.cells = self.apply_3d_life2_rules(self.cells, neighbors, self.phi)
            
            # Compute velocity from phi gradient
            self.dye_velocity = -self.compute_3d_gradient(self.phi)
            
            # Add dye where cells are born or die
            cell_changes = old_cells ^ self.cells
            if self.use_cuda:
                self.dye[cell_changes, :] += cp.array([1.0, 0.8, 0.0])  # Orange for changes
                self.dye = cp.clip(self.dye, 0, 1)
                
                # Advect dye
                self.dye = self.advect_3d_dye(self.dye, self.dye_velocity * 10.0)
                self.dye *= 0.98  # Decay
            else:
                self.dye[cell_changes, :] += np.array([1.0, 0.8, 0.0])  # Orange for changes
                self.dye = np.clip(self.dye, 0, 1)
                
                # Advect dye
                self.dye = self.advect_3d_dye(self.dye, self.dye_velocity * 10.0)
                self.dye *= 0.98  # Decay
        else:
            # Standard 3D Conway's Life
            old_cells = self.cells.copy()
            self.cells = self.apply_3d_life_rules(self.cells, neighbors)
        
        # Convergence tracking
        if self.use_cuda:
            delta = float(cp.sum(cp.abs(old_cells.astype(cp.float32) - self.cells.astype(cp.float32))))
        else:
            delta = float(np.sum(np.abs(old_cells.astype(np.float32) - self.cells.astype(np.float32))))
        
        self.current_delta = delta / (self.width * self.height * self.depth)
        self.convergence_history.append(self.current_delta)
        if len(self.convergence_history) > 10:
            self.convergence_history.pop(0)
        
        self.is_converged = len(self.convergence_history) >= 10 and np.mean(self.convergence_history) < 0.001
        self.step_count += 1
        
        # Convert to CPU arrays for visualization
        if self.use_cuda:
            cells_cpu = cp.asnumpy(self.cells)
            phi_cpu = cp.asnumpy(self.phi) if self.use_life3d else np.zeros_like(cells_cpu, dtype=np.float32)
            dye_cpu = cp.asnumpy(self.dye) if self.use_life3d else np.zeros(cells_cpu.shape + (3,), dtype=np.float32)
        else:
            cells_cpu = self.cells.copy()
            phi_cpu = self.phi.copy() if self.use_life3d else np.zeros_like(cells_cpu, dtype=np.float32)
            dye_cpu = self.dye.copy() if self.use_life3d else np.zeros(cells_cpu.shape + (3,), dtype=np.float32)
        
        return cells_cpu, phi_cpu, dye_cpu
    
    def reset(self):
        """Reset simulation to random initial state"""
        if self.use_cuda:
            self.cells = cp.random.random((self.depth, self.height, self.width)) > 0.7
            self.phi = cp.zeros((self.depth, self.height, self.width), dtype=cp.float32)
            self.dye = cp.zeros((self.depth, self.height, self.width, 3), dtype=cp.float32)
            if self.use_life3d:
                self.dye_velocity = cp.zeros((self.depth, self.height, self.width, 3), dtype=cp.float32)
        else:
            self.cells = np.random.random((self.depth, self.height, self.width)) > 0.7
            self.phi = np.zeros((self.depth, self.height, self.width), dtype=np.float32)
            self.dye = np.zeros((self.depth, self.height, self.width, 3), dtype=np.float32)
            if self.use_life3d:
                self.dye_velocity = np.zeros((self.depth, self.height, self.width, 3), dtype=np.float32)
        
        self.step_count = 0
        self.is_converged = False
        self.current_delta = 0.0
        self.convergence_history = []


class LifeFlux3DViewer:
    """Real-time 3D visualization using Open3D"""
    
    def __init__(self, simulator: LifeFlux3DSimulator, headless=False):
        self.simulator = simulator
        self.headless = headless
        
        if not headless:
            self.vis = o3d.visualization.Visualizer()
            try:
                self.vis.create_window(f"Life3D - {simulator.width}x{simulator.height}x{simulator.depth}", 
                                       width=1200, height=800)
                self.gui_available = True
                print("✓ GUI window created successfully")
            except Exception as e:
                print(f"⚠ GUI not available: {e}")
                print("Switching to headless mode...")
                self.headless = True
                self.gui_available = False
        else:
            self.gui_available = False
        
        # Visualization state
        self.playing = False
        self.show_cells = True
        self.show_dye = True
        self.show_phi = False
        self.volume_threshold = 0.1
        
        # Create initial geometry
        self.cell_cloud = o3d.geometry.PointCloud()
        self.dye_cloud = o3d.geometry.PointCloud() 
        self.phi_cloud = o3d.geometry.PointCloud()
        
        if self.gui_available:
            self.vis.add_geometry(self.cell_cloud)
            if simulator.use_life3d:
                self.vis.add_geometry(self.dye_cloud)
                self.vis.add_geometry(self.phi_cloud)
        
        if self.gui_available:
            # Set up camera and rendering options
            ctr = self.vis.get_view_control()
            ctr.set_lookat([simulator.width/2, simulator.height/2, simulator.depth/2])
            ctr.set_up([0, 1, 0])
            ctr.set_front([1, 1, 1])
            ctr.set_zoom(0.8)
            
            # Set point size for better visibility
            render_opt = self.vis.get_render_option()
            render_opt.point_size = 3.0
            render_opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
            
            # Register callbacks (try different methods for different Open3D versions)
            try:
                self.vis.register_key_callback(ord(' '), self.toggle_play)
                self.vis.register_key_callback(ord('R'), self.reset_sim)
                self.vis.register_key_callback(ord('C'), self.toggle_cells)
                self.vis.register_key_callback(ord('D'), self.toggle_dye)
                self.vis.register_key_callback(ord('P'), self.toggle_phi)
                self.vis.register_key_callback(ord('='), self.increase_threshold)
                self.vis.register_key_callback(ord('-'), self.decrease_threshold)
                self.has_callbacks = True
            except:
                # Fallback for older Open3D versions
                self.has_callbacks = False
        else:
            self.has_callbacks = False
        
        # Take a few initial steps to get stable patterns
        print("Initializing 3D patterns...")
        for i in range(3):
            cells, phi, dye = self.simulator.step()
            print(f"Step {i+1}: {cells.sum()} live cells")
        
        # Initial visualization setup
        if self.gui_available:
            self.setup_initial_geometry()
            print("✓ Initial geometry configured")
    
    def setup_initial_geometry(self):
        """Set up initial geometry with current simulation state"""
        cells, phi, dye = self.simulator.step()
        
        # Setup cells
        cell_points, cell_colors = self.extract_points_and_colors(cells)
        if len(cell_points) > 0:
            self.cell_cloud.points = o3d.utility.Vector3dVector(cell_points)
            self.cell_cloud.colors = o3d.utility.Vector3dVector(cell_colors)
            print(f"✓ Loaded {len(cell_points)} cell points")
        
        # Setup dye field if in Life3D mode
        if self.simulator.use_life3d and self.show_dye:
            dye_points, dye_colors = self.extract_points_and_colors(None, dye, self.volume_threshold)
            if len(dye_points) > 0:
                self.dye_cloud.points = o3d.utility.Vector3dVector(dye_points)
                self.dye_cloud.colors = o3d.utility.Vector3dVector(dye_colors)
                print(f"✓ Loaded {len(dye_points)} dye points")
        
        # Force geometry update and initial render
        self.vis.update_geometry(self.cell_cloud)
        if self.simulator.use_life3d:
            self.vis.update_geometry(self.dye_cloud)
            self.vis.update_geometry(self.phi_cloud)
        
        # Reset camera to ensure good view
        self.reset_camera()
        
        # Force initial render
        self.vis.update_renderer()
        print("✓ Initial render complete")
    
    def reset_camera(self):
        """Reset camera to optimal viewing position"""
        if not self.gui_available:
            return
            
        ctr = self.vis.get_view_control()
        
        # Calculate bounding box of all points
        all_points = []
        if len(self.cell_cloud.points) > 0:
            all_points.extend(np.asarray(self.cell_cloud.points))
        if len(self.dye_cloud.points) > 0:
            all_points.extend(np.asarray(self.dye_cloud.points))
        
        if len(all_points) > 0:
            all_points = np.array(all_points)
            center = np.mean(all_points, axis=0)
            span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            max_span = np.max(span)
            
            print(f"✓ Camera center: {center}, span: {max_span:.1f}")
            
            ctr.set_lookat(center.tolist())
            ctr.set_up([0, 0, 1])  # Z-up orientation
            ctr.set_front([1, 1, 1])  # Diagonal view
            ctr.set_zoom(0.5)
        else:
            # Fallback to grid center
            center = [self.simulator.width/2, self.simulator.height/2, self.simulator.depth/2]
            ctr.set_lookat(center)
            ctr.set_zoom(0.3)
    
    def extract_points_and_colors(self, volume, colors=None, threshold=0.1):
        """Extract point cloud from 3D volume"""
        if colors is None:
            # Binary volume - convert indices from (z,y,x) to (x,y,z) for Open3D
            indices = np.where(volume)
            if len(indices[0]) == 0:  # No points
                return np.array([]).reshape(0,3), np.array([]).reshape(0,3)
            
            # Reorder from (z,y,x) to (x,y,z) and scale to reasonable coordinates
            points = np.column_stack([indices[2], indices[1], indices[0]]).astype(np.float32)
            point_colors = np.ones((len(points), 3)) * np.array([0.8, 0.8, 1.0])  # Light blue
        else:
            # Volume with associated colors
            intensity = np.linalg.norm(colors, axis=-1)
            indices = np.where(intensity > threshold)
            if len(indices[0]) == 0:  # No points
                return np.array([]).reshape(0,3), np.array([]).reshape(0,3)
                
            # Reorder from (z,y,x) to (x,y,z)
            points = np.column_stack([indices[2], indices[1], indices[0]]).astype(np.float32)
            point_colors = colors[indices]
            point_colors = np.clip(point_colors, 0, 1)  # Ensure valid color range
        
        return points, point_colors
    
    def update_visualization(self):
        """Update 3D visualization with current simulation state"""
        cells, phi, dye = self.simulator.step()
        
        # Update cell visualization
        if self.show_cells:
            cell_points, cell_colors = self.extract_points_and_colors(cells)
            if len(cell_points) > 0:
                self.cell_cloud.points = o3d.utility.Vector3dVector(cell_points)
                self.cell_cloud.colors = o3d.utility.Vector3dVector(cell_colors)
                # Force the point cloud to have proper normals for rendering
                self.cell_cloud.estimate_normals()
            else:
                # Keep a minimal point to avoid empty geometry issues
                self.cell_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
                self.cell_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
        else:
            self.cell_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
            self.cell_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
        
        # Update dye visualization
        if self.simulator.use_life3d and self.show_dye:
            dye_points, dye_colors = self.extract_points_and_colors(None, dye, self.volume_threshold)
            if len(dye_points) > 0:
                self.dye_cloud.points = o3d.utility.Vector3dVector(dye_points)
                self.dye_cloud.colors = o3d.utility.Vector3dVector(dye_colors)
                self.dye_cloud.estimate_normals()
            else:
                self.dye_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
                self.dye_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
        else:
            self.dye_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
            self.dye_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
        
        # Update phi visualization  
        if self.simulator.use_life3d and self.show_phi:
            phi_normalized = (phi - phi.min()) / (phi.max() - phi.min() + 1e-8)
            phi_colors = np.zeros(phi.shape + (3,))
            phi_colors[:,:,:,2] = phi_normalized  # Blue channel for phi
            
            phi_points, phi_point_colors = self.extract_points_and_colors(None, phi_colors, self.volume_threshold)
            if len(phi_points) > 0:
                self.phi_cloud.points = o3d.utility.Vector3dVector(phi_points)
                self.phi_cloud.colors = o3d.utility.Vector3dVector(phi_point_colors)
                self.phi_cloud.estimate_normals()
            else:
                self.phi_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
                self.phi_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
        else:
            self.phi_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
            self.phi_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32))
    
    def toggle_play(self, vis):
        """Toggle play/pause"""
        self.playing = not self.playing
        return False
    
    def reset_sim(self, vis):
        """Reset simulation"""
        self.simulator.reset()
        self.update_visualization()
        return False
    
    def toggle_cells(self, vis):
        """Toggle cell visibility"""
        self.show_cells = not self.show_cells
        return False
    
    def toggle_dye(self, vis):
        """Toggle dye visibility"""
        self.show_dye = not self.show_dye
        return False
    
    def toggle_phi(self, vis):
        """Toggle phi field visibility"""
        self.show_phi = not self.show_phi
        return False
    
    def increase_threshold(self, vis):
        """Increase volume threshold"""
        self.volume_threshold = min(1.0, self.volume_threshold + 0.05)
        return False
    
    def decrease_threshold(self, vis):
        """Decrease volume threshold"""
        self.volume_threshold = max(0.01, self.volume_threshold - 0.05)
        return False
    
    def run_headless(self, max_steps=100):
        """Run in headless mode and export snapshots"""
        print(f"Running Life3D in headless mode for {max_steps} steps...")
        
        for step in range(max_steps):
            cells, phi, dye = self.simulator.step()
            
            if step % 10 == 0:
                # Create and export point cloud
                cell_points, cell_colors = self.extract_points_and_colors(cells)
                if len(cell_points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cell_points)
                    pcd.colors = o3d.utility.Vector3dVector(cell_colors)
                    
                    filename = f"lifeflux3d_step_{step:04d}.ply"
                    o3d.io.write_point_cloud(filename, pcd)
                    print(f"Exported {filename} with {len(cell_points)} points")
                
                delta_str = f"{self.simulator.current_delta:.4f}" if self.simulator.current_delta > 0 else "---"
                print(f"Step: {self.simulator.step_count}, Live cells: {cells.sum()}, Δ={delta_str}")
        
        print("✓ Headless simulation complete. Check .ply files for 3D data.")
    
    def run(self):
        """Run the 3D visualization"""
        if self.headless or not self.gui_available:
            return self.run_headless()
            
        if self.has_callbacks:
            print("Life3D Controls (Interactive):")
            print("  SPACE - Play/Pause")
            print("  R     - Reset")
            print("  C     - Toggle cells")
            print("  D     - Toggle dye field")
            print("  P     - Toggle phi field")
            print("  +/-   - Adjust volume threshold")
        else:
            print("Life3D Running (Auto-play mode - interactive controls not available)")
            self.playing = True  # Auto-start in fallback mode
        print()
        
        step_counter = 0
        try:
            while True:
                if self.playing:
                    self.update_visualization()
                    step_counter += 1
                
                # Update geometry
                if self.gui_available:
                    self.vis.update_geometry(self.cell_cloud)
                    if self.simulator.use_life3d:
                        self.vis.update_geometry(self.dye_cloud)
                        self.vis.update_geometry(self.phi_cloud)
                    
                    if not self.vis.poll_events():
                        break
                    self.vis.update_renderer()
                else:
                    break
                
                # Print status periodically
                if self.playing and step_counter % 10 == 0:
                    delta_str = f"{self.simulator.current_delta:.4f}" if self.simulator.current_delta > 0 else "---"
                    print(f"\rStep: {self.simulator.step_count}, Δ={delta_str}, Converged: {self.simulator.is_converged}", end="")
                
                # Small delay to control frame rate
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        finally:
            if self.gui_available:
                self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Life3D - 3D Cellular Automaton Visualization")
    parser.add_argument('--size', type=int, default=64, help='3D grid size (default: 64)')
    parser.add_argument('--width', type=int, help='Grid width (overrides size)')  
    parser.add_argument('--height', type=int, help='Grid height (overrides size)')
    parser.add_argument('--depth', type=int, help='Grid depth (overrides size)')
    parser.add_argument('--life-mode', choices=['life3d', 'conway3d'], default='life3d',
                       help='Simulation mode (default: life3d)')
    parser.add_argument('--phi-influence', type=float, default=0.5,
                       help='Phi field influence (0.0-1.0, default: 0.5)')
    parser.add_argument('--kernel-sigma', type=float, default=2.0,
                       help='Gaussian kernel sigma (default: 2.0)')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA acceleration')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (export PLY files)')
    
    args = parser.parse_args()
    
    # Set dimensions
    width = args.width or args.size
    height = args.height or args.size  
    depth = args.depth or args.size
    
    # Create simulator
    simulator = LifeFlux3DSimulator(
        width=width,
        height=height,
        depth=depth,
        use_life3d=(args.life_mode == 'life3d'),
        phi_influence=args.phi_influence,
        kernel_sigma=args.kernel_sigma,
        use_cuda=args.cuda,
        seed=args.seed
    )
    
    # Create and run viewer
    viewer = LifeFlux3DViewer(simulator, headless=args.headless)
    viewer.run()


if __name__ == "__main__":
    main()