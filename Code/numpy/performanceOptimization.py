#!/usr/bin/env python3
"""
NumPy Performance Optimization Techniques

This module demonstrates various optimization techniques for NumPy operations
including vectorization, memory layout optimization, and efficient algorithms.


Version: 1.0
"""

import numpy as np
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}: {(end_time - start_time)*1000:.2f} ms")
        return result
    return wrapper

def vectorization_comparison():
    """
    Compare performance between Python loops and NumPy vectorization.
    
    Returns:
        dict: Performance comparison results
    """
    print("=== Vectorization Performance Comparison ===")
    
    # Create large arrays for testing
    size = 1000000
    a = np.random.random(size)
    b = np.random.random(size)
    
    @timing_decorator
    def python_loop_approach():
        """Traditional Python loop approach."""
        result = []
        for i in range(len(a)):
            result.append(a[i] * b[i] + np.sin(a[i]))
        return np.array(result)
    
    @timing_decorator
    def vectorized_approach():
        """NumPy vectorized approach."""
        return a * b + np.sin(a)
    
    @timing_decorator
    def list_comprehension():
        """List comprehension approach."""
        return np.array([a[i] * b[i] + np.sin(a[i]) for i in range(len(a))])
    
    # Run comparisons
    result1 = python_loop_approach()
    result2 = vectorized_approach()
    result3 = list_comprehension()
    
    # Verify results are equivalent
    print(f"Results equivalent: {np.allclose(result1, result2)}")
    
    return {'loop': result1, 'vectorized': result2, 'list_comp': result3}

def memory_layout_optimization():
    """
    Demonstrate the impact of memory layout on performance.
    
    Returns:
        dict: Memory layout performance results
    """
    print("\n=== Memory Layout Optimization ===")
    
    # Create arrays with different memory layouts
    size = (1000, 1000)
    array_c = np.random.random(size).astype(np.float64, order='C')
    array_f = np.random.random(size).astype(np.float64, order='F')
    
    print(f"C-contiguous: {array_c.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {array_f.flags['F_CONTIGUOUS']}")
    
    @timing_decorator
    def row_wise_access_c():
        """Row-wise access on C-ordered array."""
        result = 0
        for i in range(array_c.shape[0]):
            result += np.sum(array_c[i, :])
        return result
    
    @timing_decorator
    def column_wise_access_c():
        """Column-wise access on C-ordered array."""
        result = 0
        for j in range(array_c.shape[1]):
            result += np.sum(array_c[:, j])
        return result
    
    @timing_decorator
    def row_wise_access_f():
        """Row-wise access on F-ordered array."""
        result = 0
        for i in range(array_f.shape[0]):
            result += np.sum(array_f[i, :])
        return result
    
    @timing_decorator
    def column_wise_access_f():
        """Column-wise access on F-ordered array."""
        result = 0
        for j in range(array_f.shape[1]):
            result += np.sum(array_f[:, j])
        return result
    
    # Test different access patterns
    print("C-ordered array:")
    result_c_row = row_wise_access_c()
    result_c_col = column_wise_access_c()
    
    print("F-ordered array:")
    result_f_row = row_wise_access_f()
    result_f_col = column_wise_access_f()
    
    return {
        'c_array': array_c, 'f_array': array_f,
        'c_row': result_c_row, 'c_col': result_c_col,
        'f_row': result_f_row, 'f_col': result_f_col
    }

def dtype_optimization():
    """
    Demonstrate performance impact of different data types.
    
    Returns:
        dict: Data type optimization results
    """
    print("\n=== Data Type Optimization ===")
    
    size = 10000000
    
    # Create arrays with different data types
    data_float64 = np.random.random(size).astype(np.float64)
    data_float32 = data_float64.astype(np.float32)
    data_int32 = (data_float64 * 1000).astype(np.int32)
    data_int16 = (data_float64 * 100).astype(np.int16)
    
    @timing_decorator
    def compute_float64():
        return np.sum(data_float64 ** 2)
    
    @timing_decorator
    def compute_float32():
        return np.sum(data_float32 ** 2)
    
    @timing_decorator
    def compute_int32():
        return np.sum(data_int32 ** 2)
    
    @timing_decorator
    def compute_int16():
        return np.sum(data_int16 ** 2)
    
    # Memory usage comparison
    print(f"Memory usage comparison:")
    print(f"float64: {data_float64.nbytes / 1024**2:.1f} MB")
    print(f"float32: {data_float32.nbytes / 1024**2:.1f} MB")
    print(f"int32: {data_int32.nbytes / 1024**2:.1f} MB")
    print(f"int16: {data_int16.nbytes / 1024**2:.1f} MB")
    
    # Performance comparison
    result_f64 = compute_float64()
    result_f32 = compute_float32()
    result_i32 = compute_int32()
    result_i16 = compute_int16()
    
    return {
        'float64': result_f64, 'float32': result_f32,
        'int32': result_i32, 'int16': result_i16
    }

if __name__ == "__main__":
    # Run performance optimization demonstrations
    vectorization_results = vectorization_comparison()
    memory_results = memory_layout_optimization()
    dtype_results = dtype_optimization()
    
    print("\n=== Broadcasting Optimization ===")
    # Demonstrate efficient broadcasting
    matrix = np.random.random((1000, 1000))
    vector = np.random.random(1000)
    
    @timing_decorator
    def broadcasting_operation():
        """Efficient broadcasting operation."""
        return matrix + vector
    
    @timing_decorator
    def manual_operation():
        """Manual operation without broadcasting."""
        result = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            result[i, :] = matrix[i, :] + vector
        return result
    
    broadcast_result = broadcasting_operation()
    manual_result = manual_operation()
    
    print(f"Results equivalent: {np.allclose(broadcast_result, manual_result)}")
    
    print("\n=== Advanced Indexing Optimization ===")
    large_array = np.random.random((10000, 10000))
    
    @timing_decorator
    def fancy_indexing():
        """Using fancy indexing."""
        indices = np.random.randint(0, 10000, 1000)
        return large_array[indices, indices]
    
    @timing_decorator
    def boolean_indexing():
        """Using boolean indexing."""
        mask = large_array[:1000, :1000] > 0.5
        return large_array[:1000, :1000][mask]
    
    fancy_result = fancy_indexing()
    boolean_result = boolean_indexing()
    
    print(f"Fancy indexing result shape: {fancy_result.shape}")
    print(f"Boolean indexing result shape: {boolean_result.shape}")
