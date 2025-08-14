#!/usr/bin/env python3
"""
NumPy Basic Operations Example

This module demonstrates fundamental NumPy operations including array creation,
manipulation, and mathematical computations that form the foundation of 
scientific computing in Python.


Version: 1.0
"""

import numpy as np
import time

def demonstrate_array_creation():
    """
    Demonstrate various methods of creating NumPy arrays.
    
    Returns:
        dict: Dictionary containing different array types
    """
    print("=== Array Creation Examples ===")
    
    # Create arrays from lists
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Create arrays with specific functions
    zeros_arr = np.zeros((3, 3))
    ones_arr = np.ones((2, 4))
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)
    
    # Random arrays
    random_arr = np.random.random((2, 3))
    
    print(f"1D Array: {arr_1d}")
    print(f"2D Array:\n{arr_2d}")
    print(f"Zeros Array:\n{zeros_arr}")
    print(f"Range Array: {range_arr}")
    print(f"Linspace Array: {linspace_arr}")
    
    return {
        '1d': arr_1d, '2d': arr_2d, 'zeros': zeros_arr,
        'ones': ones_arr, 'range': range_arr, 'random': random_arr
    }

def demonstrate_array_operations():
    """
    Show basic mathematical operations on arrays.
    
    Returns:
        dict: Results of various operations
    """
    print("\n=== Array Operations Examples ===")
    
    # Create sample arrays
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([5, 6, 7, 8])
    
    # Basic arithmetic operations
    addition = arr1 + arr2
    multiplication = arr1 * arr2
    power = arr1 ** 2
    
    # Mathematical functions
    sqrt_arr = np.sqrt(arr1)
    exp_arr = np.exp(arr1)
    log_arr = np.log(arr1)
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    print(f"Addition: {addition}")
    print(f"Element-wise multiplication: {multiplication}")
    print(f"Power of 2: {power}")
    print(f"Square root: {sqrt_arr}")
    print(f"Exponential: {exp_arr}")
    
    return {
        'addition': addition, 'multiplication': multiplication,
        'power': power, 'sqrt': sqrt_arr, 'exp': exp_arr
    }

def demonstrate_array_manipulation():
    """
    Demonstrate array reshaping, indexing, and slicing operations.
    
    Returns:
        dict: Results of manipulation operations
    """
    print("\n=== Array Manipulation Examples ===")
    
    # Create a sample array
    arr = np.arange(12)
    
    # Reshaping
    reshaped_2d = arr.reshape(3, 4)
    reshaped_3d = arr.reshape(2, 2, 3)
    
    # Indexing and slicing
    element = reshaped_2d[1, 2]
    row = reshaped_2d[1, :]
    column = reshaped_2d[:, 2]
    subarray = reshaped_2d[0:2, 1:3]
    
    print(f"Original array: {arr}")
    print(f"Reshaped 2D:\n{reshaped_2d}")
    print(f"Element at [1,2]: {element}")
    print(f"Row 1: {row}")
    print(f"Column 2: {column}")
    print(f"Subarray [0:2, 1:3]:\n{subarray}")
    
    return {
        'original': arr, 'reshaped_2d': reshaped_2d,
        'element': element, 'row': row, 'column': column
    }

if __name__ == "__main__":
    # Run demonstrations
    arrays = demonstrate_array_creation()
    operations = demonstrate_array_operations()
    manipulations = demonstrate_array_manipulation()
    
    print("\n=== Performance Comparison ===")
    # Compare NumPy vs Python list performance
    size = 100000
    python_list = list(range(size))
    numpy_array = np.arange(size)
    
    # Python list operation
    start_time = time.time()
    python_result = [x**2 for x in python_list]
    python_time = time.time() - start_time
    
    # NumPy operation
    start_time = time.time()
    numpy_result = numpy_array ** 2
    numpy_time = time.time() - start_time
    
    print(f"Python list time: {python_time:.6f} seconds")
    print(f"NumPy array time: {numpy_time:.6f} seconds")
    print(f"NumPy is {python_time/numpy_time:.1f}x faster!")