"""
Parallel Processing with Joblib

This module demonstrates Joblib's parallel processing capabilities across
different backends and use cases. It showcases performance comparisons,
backend selection strategies, and memory optimization techniques.


"""

import time
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from math import sqrt, sin, cos
import multiprocessing as mp


def cpu_bound_task(data_size, iterations=1000):
    """
    Simulate a CPU-intensive computation task.
    
    Args:
        data_size (int): Size of data array to process
        iterations (int): Number of computational iterations
        
    Returns:
        float: Computed result
    """
    data = np.random.randn(data_size)
    
    for _ in range(iterations):
        data = np.sin(data) + np.cos(data * 0.5)
    
    return np.mean(data)


def io_bound_task(duration=0.1):
    """
    Simulate an I/O-bound task with sleep.
    
    Args:
        duration (float): Sleep duration in seconds
        
    Returns:
        float: Timestamp of completion
    """
    time.sleep(duration)
    return time.time()


def memory_intensive_task(array_size=1000000):
    """
    Process large arrays to demonstrate memory management.
    
    Args:
        array_size (int): Size of array to create and process
        
    Returns:
        tuple: Statistics of the processed array
    """
    large_array = np.random.randn(array_size)
    processed = np.fft.fft(large_array)
    
    return (np.mean(np.abs(processed)), 
            np.std(np.abs(processed)),
            array_size)


def demonstrate_backend_comparison():
    """Compare performance across different parallel backends."""
    print("=== Backend Comparison Demo ===")
    
    # Test data
    task_count = 8
    data_sizes = [10000] * task_count
    
    backends_to_test = ['loky', 'threading', 'multiprocessing']
    
    for backend in backends_to_test:
        try:
            print(f"\nTesting backend: {backend}")
            start_time = time.time()
            
            with parallel_backend(backend):
                results = Parallel(n_jobs=2)(
                    delayed(cpu_bound_task)(size, 100) 
                    for size in data_sizes
                )
            
            execution_time = time.time() - start_time
            print(f"{backend} backend time: {execution_time:.3f} seconds")
            print(f"Results sample: {results[:3]}")
            
        except Exception as e:
            print(f"Backend {backend} failed: {e}")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization with large arrays."""
    print("\n=== Memory Optimization Demo ===")
    
    # Create large arrays for processing
    array_count = 4
    array_sizes = [500000] * array_count
    
    print("Processing with memory optimization:")
    start_time = time.time()
    
    # Use memory mapping for large arrays
    results = Parallel(n_jobs=2, 
                      max_nbytes='50M',  # Enable memory mapping
                      mmap_mode='r')(
        delayed(memory_intensive_task)(size) 
        for size in array_sizes
    )
    
    execution_time = time.time() - start_time
    print(f"Memory-optimized execution time: {execution_time:.3f} seconds")
    
    for i, (mean_val, std_val, size) in enumerate(results):
        print(f"Array {i+1}: size={size}, mean={mean_val:.4f}, std={std_val:.4f}")


def demonstrate_nested_parallelism():
    """Show how to handle nested parallel operations."""
    print("\n=== Nested Parallelism Demo ===")
    
    def outer_task(task_id, subtask_count=3):
        """Outer parallel task that spawns inner parallel tasks."""
        print(f"Processing outer task {task_id}")
        
        # Inner parallel computation
        inner_results = Parallel(n_jobs=2, prefer="threads")(
            delayed(cpu_bound_task)(1000, 50) 
            for _ in range(subtask_count)
        )
        
        return np.mean(inner_results)
    
    # Outer parallel loop
    outer_results = Parallel(n_jobs=2)(
        delayed(outer_task)(i) 
        for i in range(3)
    )
    
    print(f"Nested parallelism results: {outer_results}")


def main():
    """Main demonstration function."""
    print("Joblib Parallel Processing Demonstration")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    try:
        # Compare different backends
        demonstrate_backend_comparison()
        
        # Show memory optimization
        demonstrate_memory_optimization()
        
        # Demonstrate nested parallelism
        demonstrate_nested_parallelism()
        
        print("\n=== Parallel Processing Demo Complete ===")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()