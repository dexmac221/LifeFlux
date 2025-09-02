import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import imageio.v3 as iio
import colorsys

# CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("CUDA acceleration available!")
except ImportError:
    cp = np
    CUDA_AVAILABLE = False
    print("CUDA not available, using CPU")

class LifeFlux2DSimulator:
    def __init__(self, width=100, height=100, seed=2, phi_low=0.05, phi_high=0.15, 
                 stochastic_eps=0.002, flow_gain=6.0, dt_flow=0.7, 
                 kernel_radius=6, kernel_sigma=3.0, out_dir=".", 
                 display_scale=1, interpolation='nearest', use_life2=True, use_cuda=False,
                 phi_influence=0.0, colored_dye=True):
        self.width = width
        self.height = height
        self.phi_low = phi_low
        self.phi_high = phi_high
        self.stochastic_eps = stochastic_eps
        self.flow_gain = flow_gain
        self.dt_flow = dt_flow
        self.out_dir = out_dir
        self.display_scale = display_scale
        self.interpolation = interpolation
        self.use_life2 = use_life2
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.phi_influence = phi_influence
        self.kernel_sigma = kernel_sigma
        self.colored_dye = colored_dye
        
        # Choose array library based on CUDA availability
        self.xp = cp if self.use_cuda else np
        if self.use_cuda:
            print(f"Using CUDA acceleration for {width}x{height} grid")
        
        self.rng = np.random.default_rng(seed)
        self.H = height
        self.W = width
        self.K = self.gaussian_kernel(kernel_radius, kernel_sigma)
        
        # Initialize arrays on GPU if CUDA is enabled
        grid_init = (self.rng.random((self.H, self.W)) < 0.12).astype(np.int32)
        dye_init = self.rng.random((self.H, self.W))
        
        if self.use_cuda:
            self.grid = cp.asarray(grid_init)
            self.dye = cp.asarray(dye_init)
            self.phi = cp.zeros((self.H, self.W), dtype=float)
            self.vx = cp.zeros((self.H, self.W), dtype=float)
            self.vy = cp.zeros((self.H, self.W), dtype=float)
        else:
            self.grid = grid_init
            self.dye = dye_init
            self.phi = np.zeros((self.H, self.W), dtype=float)
            self.vx = np.zeros((self.H, self.W), dtype=float)
            self.vy = np.zeros((self.H, self.W), dtype=float)
            
        self.alpha = 0.2  # smoothing factor for phi inertia
        
        # Convergence tracking
        self.prev_grid = None
        self.convergence_window = []
        self.current_delta = 0.0
        self.is_converged = False
        
        self.cells_frames = []
        self.dye_frames = []
        self.combo_frames = []
        
        self.step_count = 0
        self.running = False
        
    def neighbor_sum(self, grid):
        s = self.xp.zeros_like(grid, dtype=self.xp.int32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                s += self.xp.roll(self.xp.roll(grid, dx, axis=0), dy, axis=1)
        return s
    
    def gaussian_kernel(self, radius=4, sigma=2.0):
        ax = self.xp.arange(-radius, radius + 1)
        xx, yy = self.xp.meshgrid(ax, ax, indexing="ij")
        k = self.xp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        k /= k.sum()
        return k
    
    def blur(self, field, sigma):
        """Optimized Gaussian blur: FFT on CPU, cupyx.scipy on GPU"""
        if self.xp is np:
            # CPU path: FFT-based convolution
            import numpy.fft as fft
            H, W = field.shape
            ky = np.arange(-H//2, H//2)
            kx = np.arange(-W//2, W//2)
            KX, KY = np.meshgrid(kx, ky, indexing="xy")
            G = np.exp(-(KX**2 + KY**2)/(2*sigma**2))
            G /= G.sum()
            F = fft.fft2(field)
            return np.real(fft.ifft2(F * fft.fft2(np.fft.fftshift(G), s=field.shape)))
        else:
            # GPU path: fast cupyx.scipy Gaussian filter
            from cupyx.scipy.ndimage import gaussian_filter
            return gaussian_filter(field, sigma=sigma, mode="wrap")
    
    def grad(self, field):
        gx = self.xp.roll(field, -1, axis=0) - self.xp.roll(field, 1, axis=0)
        gy = self.xp.roll(field, -1, axis=1) - self.xp.roll(field, 1, axis=1)
        return gx * 0.5, gy * 0.5
    
    def bilinear_sample(self, D, Xb, Yb):
        """Optimized bilinear sampling with single gather per corner"""
        H, W = D.shape
        x0 = self.xp.floor(Xb).astype(self.xp.int32) % H
        y0 = self.xp.floor(Yb).astype(self.xp.int32) % W
        x1 = (x0 + 1) % H
        y1 = (y0 + 1) % W
        wx = Xb - x0
        wy = Yb - y0

        # Linear indexing for efficient gathering
        i00 = x0 * W + y0
        i10 = x1 * W + y0
        i01 = x0 * W + y1
        i11 = x1 * W + y1
        Df = D.ravel()
        
        D00 = Df[i00]
        D10 = Df[i10]
        D01 = Df[i01] 
        D11 = Df[i11]

        return (1-wx)*(1-wy)*D00 + wx*(1-wy)*D10 + (1-wx)*wy*D01 + wx*wy*D11
    
    def advect_scalar(self, D, vx, vy, dt=1.0):
        """Advect scalar field using optimized bilinear sampling"""
        h, w = D.shape
        X, Y = self.xp.meshgrid(self.xp.arange(h), self.xp.arange(w), indexing="ij")
        Xb = (X - dt * vx) % h
        Yb = (Y - dt * vy) % w
        return self.bilinear_sample(D, Xb, Yb)
    
    def scale_image(self, img):
        if self.display_scale == 1:
            return img
        
        h, w = img.shape[:2]
        new_h, new_w = h * self.display_scale, w * self.display_scale
        
        if len(img.shape) == 3:  # RGB image
            scaled = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
            for c in range(img.shape[2]):
                scaled[:,:,c] = np.repeat(np.repeat(img[:,:,c], self.display_scale, axis=0), self.display_scale, axis=1)
        else:  # Grayscale
            scaled = np.repeat(np.repeat(img, self.display_scale, axis=0), self.display_scale, axis=1)
        
        return scaled
    
    def to_cpu(self, array):
        """Convert array to CPU for visualization"""
        if self.use_cuda:
            return cp.asnumpy(array)
        return array
    
    def create_colored_dye_field(self, dye, vx, vy):
        """Create colored dye field showing flow direction and intensity"""
        # Convert to CPU for processing
        dye_cpu = self.to_cpu(dye)
        vx_cpu = self.to_cpu(vx)
        vy_cpu = self.to_cpu(vy)
        
        H, W = dye_cpu.shape
        
        # Compute flow direction (hue) and speed (saturation)
        angle = np.arctan2(vy_cpu, vx_cpu)  # [-pi, pi]
        hue = (angle + np.pi) / (2*np.pi)  # [0, 1]
        
        speed = np.sqrt(vx_cpu**2 + vy_cpu**2)
        speed_max = speed.max()
        if speed_max > 0:
            sat = np.clip(speed / speed_max, 0, 1)
        else:
            sat = np.zeros_like(speed)
        
        # Use dye concentration for brightness (value)
        dye_norm = (dye_cpu - dye_cpu.min()) / (dye_cpu.max() - dye_cpu.min() + 1e-8)
        value = 0.3 + 0.7 * dye_norm  # Range [0.3, 1.0] for better visibility
        
        # Fast vectorized HSV to RGB conversion
        c = value * sat
        x = c * (1 - np.abs((hue * 6) % 2 - 1))
        m = value - c
        
        # Initialize RGB arrays
        r = np.zeros_like(hue)
        g = np.zeros_like(hue)
        b = np.zeros_like(hue)
        
        # Convert hue sectors to RGB
        h_i = np.floor(hue * 6).astype(int) % 6
        
        r = np.where(h_i == 0, c, r)
        r = np.where(h_i == 1, x, r)  
        r = np.where(h_i == 4, x, r)
        r = np.where(h_i == 5, c, r)
        
        g = np.where(h_i == 0, x, g)
        g = np.where(h_i == 1, c, g)
        g = np.where(h_i == 2, c, g)
        g = np.where(h_i == 3, x, g)
        
        b = np.where(h_i == 2, x, b)
        b = np.where(h_i == 3, c, b)
        b = np.where(h_i == 4, c, b)
        b = np.where(h_i == 5, x, b)
        
        # Add offset and convert to uint8
        r = ((r + m) * 255).astype(np.uint8)
        g = ((g + m) * 255).astype(np.uint8)
        b = ((b + m) * 255).astype(np.uint8)
        
        return np.stack([r, g, b], axis=-1)
    
    def logistic(self, x, k=25.0, x0=0.08):
        """Smooth sigmoid function for probability boosts"""
        return 1.0 / (1.0 + self.xp.exp(-k * (x - x0)))
    
    def step(self):
        if self.use_life2:
            # Life 2.0 rules with improved convergence
            mass = self.grid.astype(float)
            
            # 1. Add inertia to potential field  
            raw_phi = self.blur(mass, self.kernel_sigma)
            self.phi = (1 - self.alpha) * self.phi + self.alpha * raw_phi
            
            gx, gy = self.grad(self.phi)
            self.vx = -gy * self.flow_gain  # Store for visualization
            self.vy = gx * self.flow_gain   # Store for visualization
            
            self.dye = self.advect_scalar(self.dye, self.vx, self.vy, dt=self.dt_flow)
            
            neigh = self.neighbor_sum(self.grid)
            
            # Conway's Life rules with optional phi influence
            # Birth: exactly 3 neighbors
            born = (neigh == 3) & (self.grid == 0)
            # Survival: 2 or 3 neighbors  
            survive = ((neigh == 2) | (neigh == 3)) & (self.grid == 1)
            
            # Optional field influence for blendable rules
            if self.phi_influence > 0:
                p_survive = self.logistic(self.phi, k=25.0, x0=self.phi_low)
                # Convert to CPU for random sampling if needed
                if self.use_cuda:
                    p_survive_cpu = cp.asnumpy(p_survive)
                    mask = self.rng.random(self.phi.shape) < (self.phi_influence * p_survive_cpu)
                    mask = cp.asarray(mask)
                else:
                    mask = self.rng.random(self.phi.shape) < (self.phi_influence * p_survive)
                survive = survive | (mask & (self.grid == 1))
            
            self.grid = self.xp.where(born | survive, 1, 0)
            
            # Field only affects dye visualization, not the Life rules
        else:
            # Standard Conway's Game of Life rules
            neigh = self.neighbor_sum(self.grid)
            # Birth: exactly 3 neighbors
            born = (neigh == 3) & (self.grid == 0)
            # Survival: 2 or 3 neighbors
            survive = ((neigh == 2) | (neigh == 3)) & (self.grid == 1)
            self.grid = self.xp.where(born | survive, 1, 0)
            
            # Set zero velocity for standard Life (no flow)
            self.vx = self.xp.zeros_like(self.grid, dtype=float)
            self.vy = self.xp.zeros_like(self.grid, dtype=float)
        
        # Convergence tracking
        if self.prev_grid is not None:
            self.current_delta = float(self.xp.mean(self.xp.abs(self.grid - self.prev_grid)))
            self.convergence_window.append(self.current_delta)
            if len(self.convergence_window) > 50:
                self.convergence_window.pop(0)
            if len(self.convergence_window) == 50:
                max_delta = max(self.convergence_window)
                self.is_converged = max_delta < 0.002
                if self.is_converged and not hasattr(self, '_convergence_printed'):
                    print(f"Converged at step {self.step_count} (Δ={max_delta:.5f})")
                    self._convergence_printed = True
        
        self.prev_grid = self.grid.copy()
        self.step_count += 1
        
        # Convert to CPU for visualization
        grid_cpu = self.to_cpu(self.grid)
        cell_img = (grid_cpu * 255).astype(np.uint8)
        cell_rgb = np.stack([cell_img] * 3, axis=-1)
        
        if self.use_life2:
            if self.colored_dye:
                # Create colored dye field showing flow direction and intensity
                dye_rgb = self.create_colored_dye_field(self.dye, self.vx, self.vy)
            else:
                # Grayscale dye field (original)
                dye_cpu = self.to_cpu(self.dye)
                dnorm = (dye_cpu - dye_cpu.min()) / (dye_cpu.max() - dye_cpu.min() + 1e-8)
                dye_img = (dnorm * 255).astype(np.uint8)
                dye_rgb = np.stack([dye_img] * 3, axis=-1)
            
            combo = dye_rgb.copy()
            combo[:, :, 0] = np.maximum(combo[:, :, 0], grid_cpu * 255)
            combo[:, :, 1] = np.where(grid_cpu == 1, (combo[:, :, 1] // 2), combo[:, :, 1])
            combo[:, :, 2] = np.where(grid_cpu == 1, (combo[:, :, 2] // 2), combo[:, :, 2])
        else:
            # For standard Life, create a simple black background for dye view
            dye_rgb = np.zeros_like(cell_rgb)
            combo = cell_rgb.copy()
        
        if self.running:
            self.cells_frames.append(cell_rgb)
            self.dye_frames.append(dye_rgb)
            self.combo_frames.append(combo)
        
        cell_rgb_scaled = self.scale_image(cell_rgb)
        dye_rgb_scaled = self.scale_image(dye_rgb)
        combo_scaled = self.scale_image(combo)
        
        return cell_rgb_scaled, dye_rgb_scaled, combo_scaled
    
    def reset(self):
        grid_init = (self.rng.random((self.H, self.W)) < 0.12).astype(np.int32)
        dye_init = self.rng.random((self.H, self.W))
        
        if self.use_cuda:
            self.grid = cp.asarray(grid_init)
            self.dye = cp.asarray(dye_init)
            self.phi = cp.zeros((self.H, self.W), dtype=float)
            self.vx = cp.zeros((self.H, self.W), dtype=float)
            self.vy = cp.zeros((self.H, self.W), dtype=float)
        else:
            self.grid = grid_init
            self.dye = dye_init
            self.phi = np.zeros((self.H, self.W), dtype=float)
            self.vx = np.zeros((self.H, self.W), dtype=float)
            self.vy = np.zeros((self.H, self.W), dtype=float)
            
        self.prev_grid = None
        self.convergence_window = []
        self.step_count = 0
        self.cells_frames = []
        self.dye_frames = []
        self.combo_frames = []
    
    def save_gifs(self):
        if not self.cells_frames:
            print("No frames recorded. Start recording first.")
            return
            
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Saving {len(self.cells_frames)} frames to {self.out_dir}...")
        iio.imwrite(os.path.join(self.out_dir, "lifeflux2d_cells.gif"), self.cells_frames, duration=0.06, loop=0)
        iio.imwrite(os.path.join(self.out_dir, "lifeflux2d_dye.gif"), self.dye_frames, duration=0.06, loop=0)
        iio.imwrite(os.path.join(self.out_dir, "lifeflux2d_combo.gif"), self.combo_frames, duration=0.06, loop=0)
        
        # Create side-by-side GIFs if both cells and dye frames exist
        if self.use_life2 and len(self.cells_frames) == len(self.dye_frames):
            self.create_side_by_side_gif()
            
        print("GIFs saved successfully!")
    
    def create_side_by_side_gif(self):
        """Create a side-by-side GIF with cells on left and flow visualization on right"""
        print("Creating side-by-side demo GIF...")
        
        side_by_side_frames = []
        for cell_frame, dye_frame in zip(self.cells_frames, self.dye_frames):
            # Ensure both frames have the same height
            h1, w1 = cell_frame.shape[:2]
            h2, w2 = dye_frame.shape[:2]
            
            # Resize if needed to match height
            if h1 != h2:
                target_height = min(h1, h2)
                if len(cell_frame.shape) == 3:
                    cell_frame = cell_frame[:target_height, :, :]
                else:
                    cell_frame = cell_frame[:target_height, :]
                if len(dye_frame.shape) == 3:
                    dye_frame = dye_frame[:target_height, :, :]
                else:
                    dye_frame = dye_frame[:target_height, :]
            
            # Ensure both are RGB (3 channels)
            if len(cell_frame.shape) == 2:
                cell_frame = np.stack([cell_frame] * 3, axis=-1)
            if len(dye_frame.shape) == 2:
                dye_frame = np.stack([dye_frame] * 3, axis=-1)
            
            # Create side-by-side frame with a small separator
            separator_width = 4
            separator = np.ones((cell_frame.shape[0], separator_width, 3), dtype=np.uint8) * 128
            
            # Concatenate horizontally: cells | separator | dye
            combined_frame = np.concatenate([cell_frame, separator, dye_frame], axis=1)
            side_by_side_frames.append(combined_frame)
        
        # Save the side-by-side GIF
        side_by_side_path = os.path.join(self.out_dir, "lifeflux2d_side_by_side.gif")
        iio.imwrite(side_by_side_path, side_by_side_frames, duration=0.08, loop=0)
        print(f"Side-by-side GIF saved: {side_by_side_path}")
        
        # Also create a demo version with title overlay for README
        self.create_demo_gif_with_titles(side_by_side_frames)
    
    def create_demo_gif_with_titles(self, side_by_side_frames):
        """Create a demo GIF with titles for the README"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib import font_manager
            
            demo_frames = []
            
            # Add title overlay to each frame
            for i, frame in enumerate(side_by_side_frames):
                fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
                ax.imshow(frame)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add titles
                frame_height, frame_width = frame.shape[:2]
                cell_width = (frame_width - 4) // 2  # Account for separator
                
                # Left title: "Cellular Life"
                ax.text(cell_width//2, -20, 'Cellular Life', 
                       fontsize=16, fontweight='bold', color='white',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                # Right title: "Flow Dynamics"
                ax.text(cell_width + 4 + cell_width//2, -20, 'Flow Dynamics', 
                       fontsize=16, fontweight='bold', color='white',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                # Add step counter
                ax.text(frame_width//2, frame_height + 10, f'Step: {i+1}', 
                       fontsize=12, color='white', ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))
                
                # Add project title at the top
                ax.text(frame_width//2, -50, 'LifeFlux - Advanced Cellular Automaton', 
                       fontsize=20, fontweight='bold', color='cyan',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='darkblue', alpha=0.8))
                
                ax.set_xlim(0, frame_width)
                ax.set_ylim(frame_height + 20, -60)
                
                # Convert matplotlib figure to numpy array
                fig.canvas.draw()
                # Use buffer_rgba instead of tostring_rgb for compatibility
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert RGBA to RGB
                buf_rgb = buf[:, :, :3]
                demo_frames.append(buf_rgb)
                plt.close(fig)
            
            # Save demo GIF
            demo_path = os.path.join(self.out_dir, "lifeflux2d_demo.gif")
            iio.imwrite(demo_path, demo_frames, duration=0.1, loop=0)
            print(f"Demo GIF with titles saved: {demo_path}")
            
        except Exception as e:
            print(f"Could not create demo GIF with titles: {e}")
            # Fall back to simple version
            demo_path = os.path.join(self.out_dir, "lifeflux2d_demo.gif")
            iio.imwrite(demo_path, side_by_side_frames, duration=0.1, loop=0)
            print(f"Simple demo GIF saved: {demo_path}")


class RealtimeViewer:
    def __init__(self, simulator):
        self.simulator = simulator
        mode_name = "Life 2.0" if simulator.use_life2 else "Conway's Life"
        
        # Visualization modes
        self.view_modes = ["cells+dye", "cells_only", "dye_only"]
        self.current_view_mode = "cells+dye" if simulator.use_life2 else "cells_only"
        self.dye_color_mode = "colored"  # "colored" or "grayscale"
        
        if simulator.use_life2:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
            self.axes[0].set_title('Cells')
            self.axes[1].set_title('Dye Field')
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=(8, 6))
            self.axes = [self.axes]  # Make it a list for consistency
            self.axes[0].set_title('Cells')
        
        self.fig.suptitle(f'{mode_name} - Realtime View ({simulator.width}x{simulator.height} matrix)')
        
        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        cell_rgb, dye_rgb, combo = self.simulator.step()
        # Cache initial frames
        self._last_cell_rgb = cell_rgb
        self._last_dye_rgb = dye_rgb
        self._last_combo = combo
        
        self.im1 = self.axes[0].imshow(cell_rgb)
        if self.simulator.use_life2:
            self.im2 = self.axes[1].imshow(dye_rgb)
        else:
            self.im2 = None
        
        plt.subplots_adjust(bottom=0.2)
        
        # Move text to avoid button overlap
        self.step_text = self.fig.text(0.02, 0.18, f'Step: {self.simulator.step_count}', fontsize=10)
        self.speed_text = self.fig.text(0.02, 0.15, f'Speed: 50ms', fontsize=10)
        self.convergence_text = self.fig.text(0.02, 0.12, f'Δ=---, converged: False', fontsize=9)
        
        # Enhanced button layout with visualization controls
        ax_play = plt.axes([0.02, 0.05, 0.05, 0.04])
        ax_pause = plt.axes([0.08, 0.05, 0.05, 0.04])
        ax_reset = plt.axes([0.14, 0.05, 0.05, 0.04])
        ax_record = plt.axes([0.20, 0.05, 0.05, 0.04])
        ax_save = plt.axes([0.26, 0.05, 0.05, 0.04])
        ax_speed_down = plt.axes([0.33, 0.05, 0.03, 0.04])
        ax_speed_up = plt.axes([0.37, 0.05, 0.03, 0.04])
        
        # Visualization mode buttons
        ax_view_mode = plt.axes([0.42, 0.05, 0.06, 0.04])
        ax_color_mode = plt.axes([0.49, 0.05, 0.06, 0.04])
        ax_phi_control = plt.axes([0.56, 0.05, 0.06, 0.04])
        
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_record = Button(ax_record, 'Record')
        self.btn_save = Button(ax_save, 'Save GIFs')
        self.btn_speed_down = Button(ax_speed_down, '-')
        self.btn_speed_up = Button(ax_speed_up, '+')
        
        # New visualization buttons
        self.btn_view_mode = Button(ax_view_mode, 'View')
        self.btn_color_mode = Button(ax_color_mode, 'Color')
        self.btn_phi_control = Button(ax_phi_control, 'Phi')
        
        self.btn_play.on_clicked(self.play)
        self.btn_pause.on_clicked(self.pause)
        self.btn_reset.on_clicked(self.reset)
        self.btn_record.on_clicked(self.toggle_recording)
        self.btn_save.on_clicked(self.save_gifs)
        self.btn_speed_down.on_clicked(self.speed_down)
        self.btn_speed_up.on_clicked(self.speed_up)
        self.btn_view_mode.on_clicked(self.cycle_view_mode)
        self.btn_color_mode.on_clicked(self.toggle_color_mode)
        self.btn_phi_control.on_clicked(self.adjust_phi_influence)
        
        self.speed_interval = 50  # milliseconds
        self.animation = animation.FuncAnimation(self.fig, self.update, interval=self.speed_interval, blit=False, cache_frame_data=False)
        self.playing = False
        
    def update(self, frame):
        if self.playing:
            cell_rgb, dye_rgb, combo = self.simulator.step()
            # Cache frames for immediate button updates
            self._last_cell_rgb = cell_rgb
            self._last_dye_rgb = dye_rgb
            self._last_combo = combo
            
            self._update_displays(cell_rgb, dye_rgb, combo)
            self.step_text.set_text(f'Step: {self.simulator.step_count} | Phi: {self.simulator.phi_influence:.1f} | Recording: {self.simulator.running}')
            self.speed_text.set_text(f'Speed: {self.speed_interval}ms')
            delta_str = f"{self.simulator.current_delta:.4f}" if self.simulator.current_delta > 0 else "---"
            self.convergence_text.set_text(f'Δ={delta_str}, converged: {self.simulator.is_converged}')
        
        return_list = [self.im1]
        if self.simulator.use_life2 and self.im2:
            return_list.append(self.im2)
        return return_list
    
    def _update_displays(self, cell_rgb, dye_rgb, combo):
        """Update displays based on current visualization mode"""
        if self.current_view_mode == "cells_only":
            self.im1.set_array(cell_rgb)
            if self.im2:
                self.im2.set_array(cell_rgb)  # Show cells in both panels
        elif self.current_view_mode == "dye_only":
            display_dye = self._get_dye_display(dye_rgb)
            self.im1.set_array(display_dye)
            if self.im2:
                self.im2.set_array(display_dye)  # Show dye in both panels
        else:  # cells+dye mode
            self.im1.set_array(cell_rgb)
            if self.im2:
                display_dye = self._get_dye_display(dye_rgb)
                self.im2.set_array(display_dye)
    
    def _get_dye_display(self, dye_rgb):
        """Get dye display based on color mode"""
        if self.dye_color_mode == "grayscale" and self.simulator.use_life2:
            # Create grayscale version
            gray = np.dot(dye_rgb[...,:3], [0.299, 0.587, 0.114])
            return np.stack([gray] * 3, axis=-1).astype(np.uint8)
        return dye_rgb
    
    def play(self, event):
        self.playing = True
    
    def pause(self, event):
        self.playing = False
    
    def reset(self, event):
        self.simulator.reset()
        cell_rgb, dye_rgb, combo = self.simulator.step()
        # Cache frames
        self._last_cell_rgb = cell_rgb
        self._last_dye_rgb = dye_rgb
        self._last_combo = combo
        
        self._update_displays(cell_rgb, dye_rgb, combo)
        self.step_text.set_text(f'Step: {self.simulator.step_count} | Recording: {self.simulator.running}')
        self.speed_text.set_text(f'Speed: {self.speed_interval}ms')
        self.convergence_text.set_text(f'Δ=---, converged: False')
        self.fig.canvas.draw()
    
    def toggle_recording(self, event):
        self.simulator.running = not self.simulator.running
        if self.simulator.running:
            print("Started recording frames...")
        else:
            print("Stopped recording frames.")
    
    def save_gifs(self, event):
        self.simulator.save_gifs()
    
    def speed_up(self, event):
        self.speed_interval = max(10, self.speed_interval - 20)
        # Force animation to restart with new interval
        self.animation.event_source.stop()
        self.animation = animation.FuncAnimation(
            self.fig, self.update, interval=self.speed_interval, 
            blit=False, cache_frame_data=False
        )
        if self.playing:
            self.animation.event_source.start()
        self.speed_text.set_text(f'Speed: {self.speed_interval}ms')
        self.fig.canvas.draw_idle()
        print(f"Speed increased: {self.speed_interval}ms")
    
    def speed_down(self, event):
        self.speed_interval = min(1000, self.speed_interval + 50)
        # Force animation to restart with new interval
        self.animation.event_source.stop()
        self.animation = animation.FuncAnimation(
            self.fig, self.update, interval=self.speed_interval, 
            blit=False, cache_frame_data=False
        )
        if self.playing:
            self.animation.event_source.start()
        self.speed_text.set_text(f'Speed: {self.speed_interval}ms')
        self.fig.canvas.draw_idle()
        print(f"Speed decreased: {self.speed_interval}ms")
    
    def cycle_view_mode(self, event):
        """Cycle through visualization modes: cells+dye -> cells_only -> dye_only"""
        current_idx = self.view_modes.index(self.current_view_mode)
        next_idx = (current_idx + 1) % len(self.view_modes)
        self.current_view_mode = self.view_modes[next_idx]
        
        # Update panel titles
        if self.current_view_mode == "cells_only":
            self.axes[0].set_title('Cells Only')
            if self.simulator.use_life2:
                self.axes[1].set_title('Cells Only')
        elif self.current_view_mode == "dye_only":
            self.axes[0].set_title('Dye Field Only')
            if self.simulator.use_life2:
                self.axes[1].set_title('Dye Field Only')
        else:  # cells+dye
            self.axes[0].set_title('Cells')
            if self.simulator.use_life2:
                self.axes[1].set_title('Dye Field')
        
        # Force immediate update
        if hasattr(self, '_last_cell_rgb'):
            self._update_displays(self._last_cell_rgb, self._last_dye_rgb, self._last_combo)
            self.fig.canvas.draw()
    
    def toggle_color_mode(self, event):
        """Toggle between colored and grayscale dye field"""
        if self.dye_color_mode == "colored":
            self.dye_color_mode = "grayscale"
        else:
            self.dye_color_mode = "colored"
        
        # Force immediate update
        if hasattr(self, '_last_cell_rgb'):
            self._update_displays(self._last_cell_rgb, self._last_dye_rgb, self._last_combo)
            self.fig.canvas.draw()
    
    def adjust_phi_influence(self, event):
        """Cycle through phi influence values"""
        phi_values = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
        current_idx = 0
        
        # Find current value index
        for i, val in enumerate(phi_values):
            if abs(self.simulator.phi_influence - val) < 0.01:
                current_idx = i
                break
        
        # Move to next value
        new_idx = (current_idx + 1) % len(phi_values)
        self.simulator.phi_influence = phi_values[new_idx]
        
        print(f"Phi influence: {self.simulator.phi_influence:.1f}")
        
        # Update status text to show phi value
        self.step_text.set_text(f'Step: {self.simulator.step_count} | Phi: {self.simulator.phi_influence:.1f}')
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()


def run_batch_mode():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=100, help="Game matrix width")
    p.add_argument("--height", type=int, default=100, help="Game matrix height")
    p.add_argument("--matrix", type=str, help="Game matrix as WIDTHxHEIGHT (e.g., 100x100)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--phi-low", type=float, default=0.05)
    p.add_argument("--phi-high", type=float, default=0.15)
    p.add_argument("--stochastic-eps", type=float, default=0.002)
    p.add_argument("--flow-gain", type=float, default=6.0)
    p.add_argument("--dt-flow", type=float, default=0.7)
    p.add_argument("--kernel-radius", type=int, default=6)
    p.add_argument("--kernel-sigma", type=float, default=3.0)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--display-scale", type=int, default=1, help="Display resolution scale factor (1=original, 2=2x, 4=4x, etc)")
    p.add_argument("--interpolation", type=str, default="nearest", choices=["nearest", "linear"], help="Scaling interpolation method")
    p.add_argument("--life-mode", type=str, default="life2", choices=["standard", "life2"], help="Life rules: 'standard' for Conway's Life, 'life2' for Life 2.0 with fields")
    p.add_argument("--cuda", action="store_true", help="Use CUDA acceleration for giant grids (requires cupy)")
    p.add_argument("--phi-influence", type=float, default=0.0, help="Field influence on Life rules (0.0=pure Conway, 0.2=gentle bias, 1.0=full Life2)")
    p.add_argument("--colored-dye", action="store_true", default=True, help="Use colored dye field (default: True)")
    p.add_argument("--grayscale-dye", action="store_true", help="Use grayscale dye field instead of colored")
    p.add_argument("--demo", action="store_true", help="Generate optimized demo GIFs with enhanced side-by-side visualization")
    args = p.parse_args()
    
    # Parse matrix dimensions if provided
    width, height = args.width, args.height
    if args.matrix:
        try:
            w_str, h_str = args.matrix.split('x')
            width, height = int(w_str), int(h_str)
        except ValueError:
            print(f"Invalid matrix format: {args.matrix}. Use WIDTHxHEIGHT (e.g., 100x100)")
            return
    
    # Apply demo mode optimizations if requested
    if args.demo:
        # Override settings for better demo visuals
        if args.life_mode == "life2":
            args.phi_influence = max(args.phi_influence, 0.3)  # Ensure visible field effects
            args.flow_gain = max(args.flow_gain, 8.0)          # Ensure visible flow
            args.colored_dye = True                             # Force colored dye for demos
        
        print("🎬 Demo mode: Optimizing settings for visual appeal")
        print(f"   Phi influence: {args.phi_influence}")
        print(f"   Flow gain: {args.flow_gain}")
        print(f"   Colored dye: {not args.grayscale_dye}")
    
    simulator = LifeFlux2DSimulator(
        width=width, height=height, seed=args.seed,
        phi_low=args.phi_low, phi_high=args.phi_high,
        stochastic_eps=args.stochastic_eps,
        flow_gain=args.flow_gain, dt_flow=args.dt_flow,
        kernel_radius=args.kernel_radius, kernel_sigma=args.kernel_sigma,
        out_dir=args.out_dir,
        display_scale=args.display_scale, interpolation=args.interpolation,
        use_life2=(args.life_mode == "life2"), use_cuda=args.cuda,
        phi_influence=args.phi_influence,
        colored_dye=not args.grayscale_dye
    )
    
    simulator.running = True
    
    for _ in range(args.steps):
        simulator.step()
    
    simulator.save_gifs()


def run_realtime_mode():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=100, help="Game matrix width")
    p.add_argument("--height", type=int, default=100, help="Game matrix height")
    p.add_argument("--matrix", type=str, help="Game matrix as WIDTHxHEIGHT (e.g., 100x100)")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--phi-low", type=float, default=0.05)
    p.add_argument("--phi-high", type=float, default=0.15)
    p.add_argument("--stochastic-eps", type=float, default=0.002)
    p.add_argument("--flow-gain", type=float, default=6.0)
    p.add_argument("--dt-flow", type=float, default=0.7)
    p.add_argument("--kernel-radius", type=int, default=6)
    p.add_argument("--kernel-sigma", type=float, default=3.0)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--display-scale", type=int, default=4, help="Display resolution scale factor (1=original, 2=2x, 4=4x, etc)")
    p.add_argument("--interpolation", type=str, default="nearest", choices=["nearest", "linear"], help="Scaling interpolation method")
    p.add_argument("--life-mode", type=str, default="life2", choices=["standard", "life2"], help="Life rules: 'standard' for Conway's Life, 'life2' for Life 2.0 with fields")
    p.add_argument("--cuda", action="store_true", help="Use CUDA acceleration for giant grids (requires cupy)")
    p.add_argument("--phi-influence", type=float, default=0.0, help="Field influence on Life rules (0.0=pure Conway, 0.2=gentle bias, 1.0=full Life2)")
    p.add_argument("--colored-dye", action="store_true", default=True, help="Use colored dye field (default: True)")
    p.add_argument("--grayscale-dye", action="store_true", help="Use grayscale dye field instead of colored")
    p.add_argument("--realtime", action="store_true", help="Run in realtime mode")
    args = p.parse_args()
    
    # Parse matrix dimensions if provided
    width, height = args.width, args.height
    if args.matrix:
        try:
            w_str, h_str = args.matrix.split('x')
            width, height = int(w_str), int(h_str)
        except ValueError:
            print(f"Invalid matrix format: {args.matrix}. Use WIDTHxHEIGHT (e.g., 100x100)")
            return
    
    simulator = LifeFlux2DSimulator(
        width=width, height=height, seed=args.seed,
        phi_low=args.phi_low, phi_high=args.phi_high,
        stochastic_eps=args.stochastic_eps,
        flow_gain=args.flow_gain, dt_flow=args.dt_flow,
        kernel_radius=args.kernel_radius, kernel_sigma=args.kernel_sigma,
        out_dir=args.out_dir,
        display_scale=args.display_scale, interpolation=args.interpolation,
        use_life2=(args.life_mode == "life2"), use_cuda=args.cuda,
        phi_influence=args.phi_influence,
        colored_dye=not args.grayscale_dye
    )
    viewer = RealtimeViewer(simulator)
    viewer.show()


def main():
    import sys
    if "--realtime" in sys.argv:
        run_realtime_mode()
    else:
        run_batch_mode()


if __name__ == "__main__":
    main()