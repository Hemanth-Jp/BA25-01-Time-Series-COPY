"""
Memory Caching with Joblib

This module demonstrates Joblib's memory caching capabilities for optimizing
expensive computational operations. It showcases transparent disk-caching,
cache management, and performance comparison between cached and uncached operations.


"""

import time
import numpy as np
from joblib import Memory
import os
import shutil
import tempfile


def expensive_computation(n_samples, n_features, complexity_factor=1000):
    """
    Simulate an expensive computation operation.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features per sample
        complexity_factor (int): Factor to increase computation complexity
        
    Returns:
        tuple: Mean and standard deviation of generated data
    """
    print(f"Computing expensive operation for {n_samples}x{n_features} data...")
    
    # Simulate expensive computation with sleep and complex operations
    time.sleep(0.1)  # Simulate I/O or network delay
    
    # Generate random data and perform complex computations
    data = np.random.RandomState(42).randn(n_samples, n_features)
    
    for _ in range(complexity_factor):
        data = np.sin(data) + np.cos(data) * 0.1
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    print(f"Computation completed: mean={mean_val:.4f}, std={std_val:.4f}")
    return mean_val, std_val


def get_writable_cache_dir(name):
    """Get a writable cache directory."""
    # Try multiple locations in order of preference
    potential_dirs = [
        os.path.expanduser(f"~/joblib_cache_{name}"),  # User home directory
        os.path.join(tempfile.gettempdir(), f"joblib_cache_{name}"),  # System temp directory
        tempfile.mkdtemp(prefix=f"joblib_cache_{name}_")  # Guaranteed writable temp dir
    ]
    
    for cache_dir in potential_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test if we can write to it
            test_file = os.path.join(cache_dir, "test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Using cache directory: {cache_dir}")
            return cache_dir
        except (OSError, PermissionError):
            continue
    
    raise OSError("Could not find a writable directory for caching")


def demonstrate_basic_caching():
    """Demonstrate basic memory caching functionality."""
    print("=== Basic Memory Caching Demo ===")
    
    # Setup cache directory in a writable location
    cache_dir = get_writable_cache_dir("basic")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Create Memory object with caching
    memory = Memory(location=cache_dir, verbose=1)
    
    # Create cached version of expensive function
    cached_computation = memory.cache(expensive_computation)
    
    # First call - will compute and cache
    print("\nFirst call (will cache):")
    start_time = time.time()
    result1 = cached_computation(1000, 50)
    first_call_time = time.time() - start_time
    print(f"First call time: {first_call_time:.3f} seconds")
    
    # Second call - will load from cache
    print("\nSecond call (from cache):")
    start_time = time.time()
    result2 = cached_computation(1000, 50)
    second_call_time = time.time() - start_time
    print(f"Second call time: {second_call_time:.3f} seconds")
    
    # Verify results are identical
    assert result1 == result2, "Cached results should be identical"
    print(f"Speed improvement: {first_call_time/second_call_time:.1f}x faster")
    
    return memory, cache_dir


def demonstrate_cache_management():
    """Demonstrate advanced cache management features."""
    print("\n=== Cache Management Demo ===")
    
    cache_dir = get_writable_cache_dir("advanced")
    memory = Memory(location=cache_dir, verbose=1)
    
    @memory.cache
    def matrix_operations(size, operation_type):
        """Perform various matrix operations."""
        print(f"Computing {operation_type} for {size}x{size} matrix")
        matrix = np.random.randn(size, size)
        
        if operation_type == 'eigenvalues':
            return np.linalg.eigvals(matrix)
        elif operation_type == 'svd':
            return np.linalg.svd(matrix)
        elif operation_type == 'determinant':
            return np.linalg.det(matrix)
    
    # Cache multiple operations
    print("Caching multiple operations:")
    eigen_result = matrix_operations(100, 'eigenvalues')
    det_result = matrix_operations(100, 'determinant')
    svd_result = matrix_operations(100, 'svd')
    
    # Check cache info
    print(f"\nCache location: {memory.location}")
    
    # Clear specific cache entry
    matrix_operations.clear()
    print("Cache cleared for matrix_operations")
    
    return cache_dir


def demonstrate_in_memory_caching():
    """Demonstrate in-memory caching as an alternative."""
    print("\n=== In-Memory Caching Demo ===")
    
    # Use None as location for in-memory caching
    memory = Memory(location=None, verbose=1)
    
    @memory.cache
    def quick_computation(x, y):
        """A quick computation for in-memory demo."""
        print(f"Computing {x} + {y}")
        time.sleep(0.1)  # Simulate some work
        return x + y
    
    print("First call (will compute):")
    result1 = quick_computation(5, 3)
    
    print("Second call (from memory):")
    result2 = quick_computation(5, 3)
    
    print(f"Results: {result1}, {result2}")
    print("Note: In-memory caching doesn't persist between sessions")


def main():
    """Main demonstration function."""
    cache_dirs = []
    
    try:
        # Run basic caching demo
        memory, basic_cache_dir = demonstrate_basic_caching()
        cache_dirs.append(basic_cache_dir)
        
        # Run cache management demo
        advanced_cache_dir = demonstrate_cache_management()
        cache_dirs.append(advanced_cache_dir)
        
        # Run in-memory caching demo
        demonstrate_in_memory_caching()
        
        print("\n=== Memory Caching Demo Complete ===")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        raise
    finally:
        # Cleanup cache directories
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print(f"Cleaned up cache directory: {cache_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean up {cache_dir}: {e}")


if __name__ == "__main__":
    main()