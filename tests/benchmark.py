#!/usr/bin/env python3
"""
Performance benchmarks for LifeFlow
"""

import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def benchmark_2d_performance():
    """Benchmark 2D Life performance with different configurations"""
    from lifeflux2d import LifeFlux2DSimulator
    
    print("=== 2D Life Performance Benchmarks ===")
    
    configs = [
        {"size": (50, 50), "cuda": False, "name": "Small CPU"},
        {"size": (100, 100), "cuda": False, "name": "Medium CPU"},
        {"size": (200, 200), "cuda": False, "name": "Large CPU"},
    ]
    
    # Add CUDA benchmarks if available
    try:
        import cupy
        configs.extend([
            {"size": (100, 100), "cuda": True, "name": "Medium CUDA"},
            {"size": (200, 200), "cuda": True, "name": "Large CUDA"},
            {"size": (500, 500), "cuda": True, "name": "Huge CUDA"},
        ])
    except ImportError:
        print("CUDA not available, skipping GPU benchmarks")
    
    for config in configs:
        width, height = config["size"]
        use_cuda = config["cuda"]
        name = config["name"]
        
        print(f"\nTesting {name} ({width}x{height})...")
        
        try:
            # Initialize
            start_time = time.time()
            sim = LifeFlux2DSimulator(width=width, height=height, 
                               use_life2=True, use_cuda=use_cuda, seed=42)
            init_time = time.time() - start_time
            
            # Warm up
            sim.step()
            sim.step()
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                sim.step()
            benchmark_time = time.time() - start_time
            
            steps_per_second = 10 / benchmark_time
            print(f"  Init: {init_time:.3f}s")
            print(f"  10 steps: {benchmark_time:.3f}s ({steps_per_second:.1f} steps/sec)")
            print(f"  Memory: {width * height * 4 * 4 / 1024 / 1024:.1f} MB (approx)")
            
        except Exception as e:
            print(f"  Error: {e}")

def benchmark_3d_performance():
    """Benchmark 3D Life performance"""
    try:
        from lifeflux3d import LifeFlux3DSimulator
        print("\n=== 3D Life Performance Benchmarks ===")
        
        configs = [
            {"size": (16, 16, 16), "cuda": False, "name": "Small 3D CPU"},
            {"size": (32, 32, 32), "cuda": False, "name": "Medium 3D CPU"},
        ]
        
        # Add CUDA if available
        try:
            import cupy
            configs.extend([
                {"size": (32, 32, 32), "cuda": True, "name": "Medium 3D CUDA"},
                {"size": (64, 64, 64), "cuda": True, "name": "Large 3D CUDA"},
            ])
        except ImportError:
            pass
        
        for config in configs:
            width, height, depth = config["size"]
            use_cuda = config["cuda"]
            name = config["name"]
            
            print(f"\nTesting {name} ({width}x{height}x{depth})...")
            
            try:
                start_time = time.time()
                sim = LifeFlux3DSimulator(width=width, height=height, depth=depth,
                                    use_cuda=use_cuda, seed=42)
                init_time = time.time() - start_time
                
                # Warm up
                sim.step()
                
                # Benchmark
                start_time = time.time()
                for _ in range(5):
                    sim.step()
                benchmark_time = time.time() - start_time
                
                steps_per_second = 5 / benchmark_time
                memory_mb = width * height * depth * 4 * 8 / 1024 / 1024
                
                print(f"  Init: {init_time:.3f}s")
                print(f"  5 steps: {benchmark_time:.3f}s ({steps_per_second:.1f} steps/sec)")
                print(f"  Memory: {memory_mb:.1f} MB (approx)")
                
            except Exception as e:
                print(f"  Error: {e}")
                
    except ImportError:
        print("\n3D Life module not available for benchmarking")

def benchmark_memory_usage():
    """Estimate memory usage for different grid sizes"""
    print("\n=== Memory Usage Estimates ===")
    
    sizes_2d = [(100, 100), (200, 200), (500, 500), (1000, 1000)]
    sizes_3d = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    
    print("\n2D Grid Memory Usage:")
    for w, h in sizes_2d:
        # Rough estimate: grid, dye, phi, vx, vy (int32 + 4*float32 per cell)
        memory_mb = w * h * (4 + 4*4) / 1024 / 1024
        print(f"  {w}x{h}: ~{memory_mb:.1f} MB")
    
    print("\n3D Grid Memory Usage:")
    for w, h, d in sizes_3d:
        # Rough estimate: cells, phi, dye (bool + float32 + 3*float32 per cell)
        memory_mb = w * h * d * (1 + 4 + 3*4) / 1024 / 1024
        print(f"  {w}x{h}x{d}: ~{memory_mb:.1f} MB")

if __name__ == "__main__":
    print("LifeFlow Performance Benchmarks")
    print("=" * 50)
    
    benchmark_2d_performance()
    benchmark_3d_performance() 
    benchmark_memory_usage()
    
    print("\nBenchmarks complete!")
