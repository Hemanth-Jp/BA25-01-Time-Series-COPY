#!/usr/bin/env python3
"""
NumPy Error Handling Best Practices 

This module demonstrates comprehensive error handling patterns and best
practices for robust NumPy applications including numerical stability,
shape compatibility, and data validation.


Version: 1.1 
"""

import numpy as np
import warnings
from typing import Union, Tuple, Optional
import time

class NumPyErrorHandler:
    """
    A comprehensive error handler for NumPy operations.
    
    Provides methods for safe array operations with proper error handling
    and validation.
    """
    
    @staticmethod
    def safe_array_creation(data, dtype=None, validate=True):
        """
        Safely create NumPy arrays with validation.
        
        Args:
            data: Input data for array creation
            dtype: Desired data type
            validate: Whether to perform validation
            
        Returns:
            np.ndarray: Created array
            
        Raises:
            ValueError: If data cannot be converted to array
            TypeError: If dtype is invalid
        """
        try:
            if validate and hasattr(data, '__len__') and len(data) == 0:
                raise ValueError("Cannot create array from empty data")
            
            array = np.array(data, dtype=dtype)
            
            if validate and array.size > 0:
                # Only check for inf/nan on numeric types
                if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
                    if np.any(np.isinf(array)):
                        warnings.warn("Array contains infinite values", UserWarning)
                    if np.any(np.isnan(array)):
                        warnings.warn("Array contains NaN values", UserWarning)
                        
                if array.size == 0:
                    warnings.warn("Created empty array", UserWarning)
            
            return array
            
        except ValueError as e:
            raise ValueError(f"Array creation failed: {str(e)}")
        except TypeError as e:
            raise TypeError(f"Invalid dtype specified: {str(e)}")
    
    @staticmethod
    def safe_matrix_operations(A, B, operation='multiply'):
        """
        Perform safe matrix operations with shape validation.
        
        Args:
            A: First matrix
            B: Second matrix
            operation: Type of operation ('multiply', 'add', 'subtract')
            
        Returns:
            np.ndarray: Result of operation
            
        Raises:
            ValueError: If shapes are incompatible
            LinAlgError: If operation is not supported
        """
        try:
            A = np.asarray(A, dtype=float)  # Ensure floating point for safety
            B = np.asarray(B, dtype=float)
            
            if operation == 'multiply':
                if A.ndim == 1:
                    A = A.reshape(1, -1)
                if B.ndim == 1:
                    B = B.reshape(-1, 1)
                    
                if A.shape[-1] != B.shape[-2]:
                    raise ValueError(
                        f"Cannot multiply matrices with shapes {A.shape} and {B.shape}"
                    )
                result = np.dot(A, B)
                
            elif operation in ['add', 'subtract']:
                try:
                    if operation == 'add':
                        result = A + B
                    else:
                        result = A - B
                except ValueError as e:
                    raise ValueError(
                        f"Cannot {operation} arrays with shapes {A.shape} and {B.shape}: {str(e)}"
                    )
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Check for numerical issues
            if np.any(np.isnan(result)):
                warnings.warn("Operation resulted in NaN values", RuntimeWarning)
            if np.any(np.isinf(result)):
                warnings.warn("Operation resulted in infinite values", RuntimeWarning)
            
            return result
            
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Linear algebra error: {str(e)}")
    
    @staticmethod
    def safe_division(numerator, denominator, handle_zero='mask'):
        """
        Perform safe division with zero handling.
        
        Args:
            numerator: Numerator array
            denominator: Denominator array
            handle_zero: How to handle zeros ('mask', 'inf', 'nan', 'raise')
            
        Returns:
            np.ndarray or np.ma.MaskedArray: Division result
            
        Raises:
            ZeroDivisionError: If handle_zero='raise' and zeros found
            ValueError: If handle_zero method is invalid
        """
        # Convert to float arrays to avoid casting issues
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        
        # Find zero elements (use small tolerance for floating point)
        zero_mask = np.abs(denominator) < np.finfo(float).eps * 10
        
        if np.any(zero_mask):
            if handle_zero == 'raise':
                raise ZeroDivisionError("Division by zero encountered")
            elif handle_zero == 'mask':
                # Create result array with proper dtype
                result = np.zeros_like(numerator, dtype=float)
                # Only divide where denominator is not zero
                np.divide(numerator, denominator, out=result, where=~zero_mask)
                return np.ma.masked_array(result, mask=zero_mask)
            elif handle_zero == 'inf':
                with np.errstate(divide='ignore'):
                    return numerator / denominator
            elif handle_zero == 'nan':
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = numerator / denominator
                    result[zero_mask] = np.nan
                    return result
            else:
                raise ValueError(f"Invalid handle_zero method: {handle_zero}")
        else:
            return numerator / denominator

def demonstrate_input_validation():
    """
    Demonstrate input validation techniques for NumPy functions.
    
    Returns:
        dict: Validation results
    """
    print("=== Input Validation Examples ===")
    
    handler = NumPyErrorHandler()
    
    # Test valid inputs
    try:
        valid_array = handler.safe_array_creation([1, 2, 3, 4, 5])
        print(f"Valid array created: {valid_array}")
    except (ValueError, TypeError) as e:
        print(f"Error creating valid array: {e}")
    
    # Test invalid inputs
    test_cases = [
        ([], "empty list"),
        ([1, 2, 3], "integer list"),  # This should work fine
        ([1.0, 2.0, np.inf], "infinite values"),
        ([1.0, 2.0, np.nan], "NaN values")
    ]
    
    for data, description in test_cases:
        try:
            result = handler.safe_array_creation(data, validate=True)
            print(f"Created array from {description}: {result}")
        except (ValueError, TypeError) as e:
            print(f"Failed to create array from {description}: {e}")
    
    return {"handler": handler}

def demonstrate_numerical_stability():
    """
    Show techniques for maintaining numerical stability.
    
    Returns:
        dict: Stability demonstration results
    """
    print("\n=== Numerical Stability Examples ===")
    
    # Demonstrate precision issues
    a = np.array([1e16, 1, -1e16])
    naive_sum = np.sum(a)
    
    # Use Kahan summation for better stability
    def kahan_sum(arr):
        """Implement Kahan summation for better numerical stability."""
        total = 0.0
        compensation = 0.0
        
        for value in arr:
            y = value - compensation
            temp = total + y
            compensation = (temp - total) - y
            total = temp
        return total
    
    stable_sum = kahan_sum(a)
    
    print(f"Naive sum: {naive_sum}")
    print(f"Kahan sum: {stable_sum}")
    
    # Demonstrate safe logarithm computation
    def safe_log(x, min_val=1e-15):
        """Compute logarithm with numerical stability."""
        x = np.asarray(x, dtype=float)
        x_safe = np.maximum(x, min_val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.log(x_safe)
    
    # Test with problematic values
    test_values = np.array([0.0, 1e-20, 1e-10, 1.0, 10.0])
    safe_logs = safe_log(test_values)
    
    print(f"Test values: {test_values}")
    print(f"Safe logarithms: {safe_logs}")
    
    # Demonstrate condition number checking
    def check_matrix_condition(matrix, threshold=1e12):
        """Check if matrix is well-conditioned."""
        try:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                warnings.warn("Matrix is not square, using pseudo-condition number", UserWarning)
            
            cond_num = np.linalg.cond(matrix)
            if cond_num > threshold:
                warnings.warn(
                    f"Matrix is ill-conditioned (cond={cond_num:.2e})",
                    RuntimeWarning
                )
            return cond_num
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Could not compute condition number: {e}", RuntimeWarning)
            return np.inf
    
    # Test well-conditioned matrix
    np.random.seed(42)  # For reproducible results
    well_conditioned = np.eye(3) + 0.1 * np.random.random((3, 3))
    cond1 = check_matrix_condition(well_conditioned)
    
    # Test ill-conditioned matrix
    ill_conditioned = np.array([[1.0, 1.0], [1.0, 1.0001]])
    cond2 = check_matrix_condition(ill_conditioned)
    
    print(f"Well-conditioned matrix condition: {cond1:.2e}")
    print(f"Ill-conditioned matrix condition: {cond2:.2e}")
    
    return {
        "naive_sum": naive_sum, 
        "stable_sum": stable_sum,
        "safe_logs": safe_logs, 
        "condition_numbers": [cond1, cond2]
    }

def demonstrate_error_recovery():
    """
    Show error recovery and graceful degradation techniques.
    
    Returns:
        dict: Error recovery results
    """
    print("\n=== Error Recovery Examples ===")
    
    handler = NumPyErrorHandler()
    
    # Safe division examples
    numerator = np.array([1, 2, 3, 4])
    denominator = np.array([2, 0, 1, 0])
    
    division_methods = ['mask', 'inf', 'nan']
    results = {}
    
    for method in division_methods:
        try:
            result = handler.safe_division(numerator, denominator, method)
            print(f"Division with {method} handling: {result}")
            if hasattr(result, 'mask'):
                print(f"  Masked values: {result.mask}")
            results[method] = result
        except ZeroDivisionError as e:
            print(f"Division failed with {method}: {e}")
            results[method] = None
    
    # Test raise method separately
    try:
        handler.safe_division(numerator, denominator, 'raise')
    except ZeroDivisionError as e:
        print(f"Division with raise handling: {e}")
    
    # Matrix operation error handling
    matrices = [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),  # Compatible
        (np.array([[1, 2, 3]]), np.array([[4, 5], [6, 7]])),  # Incompatible shapes
        (np.array([1, 2]), np.array([3, 4]))  # Vector case
    ]
    
    for i, (A, B) in enumerate(matrices):
        try:
            result = handler.safe_matrix_operations(A, B, 'multiply')
            print(f"Matrix multiplication {i+1}: Success, shape {result.shape}")
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Matrix multiplication {i+1}: Failed - {e}")
    
    return {"division_results": results, "matrices": matrices}

def demonstrate_context_managers():
    """
    Show how to use NumPy error state context managers.
    """
    print("\n=== Context Manager Examples ===")
    
    # Demonstrate different error handling strategies
    problematic_array = np.array([1.0, 0.0, -1.0])
    
    print("Default behavior:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings for demo
        try:
            result = np.sqrt(problematic_array)
            print(f"sqrt of {problematic_array}: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nWith error state control:")
    with np.errstate(invalid='ignore'):
        result = np.sqrt(problematic_array)
        print(f"sqrt with invalid='ignore': {result}")
    
    with np.errstate(invalid='warn'):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = np.sqrt(problematic_array)
            if w:
                print(f"Warning caught: {w[0].message}")
    
    print("\nDivision by zero handling:")
    test_division = np.array([1.0, 2.0]) / np.array([1.0, 0.0])
    print(f"Default division: {test_division}")
    
    with np.errstate(divide='ignore'):
        safe_division = np.array([1.0, 2.0]) / np.array([1.0, 0.0])
        print(f"Division with ignore: {safe_division}")

if __name__ == "__main__":
    # Set up warning handling
    warnings.simplefilter('default')
    
    try:
        # Run error handling demonstrations
        validation_results = demonstrate_input_validation()
        stability_results = demonstrate_numerical_stability()
        recovery_results = demonstrate_error_recovery()
        demonstrate_context_managers()
        
        print("\n=== Performance vs Safety Trade-offs ===")
        
        # Compare performance of safe vs unsafe operations
        np.random.seed(42)
        large_array = np.random.random(100000)  # Smaller array for demo
        small_values = large_array * 1e-20
        
        # Unsafe operation
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unsafe_log = np.log(small_values)
        unsafe_time = time.time() - start_time
        
        # Safe operation
        start_time = time.time()
        safe_log_result = np.log(np.maximum(small_values, 1e-15))
        safe_time = time.time() - start_time
        
        print(f"Unsafe log time: {unsafe_time*1000:.2f} ms")
        print(f"Safe log time: {safe_time*1000:.2f} ms")
        if unsafe_time > 0:
            print(f"Safety overhead: {((safe_time - unsafe_time)/unsafe_time)*100:.1f}%")
        
        # Count warnings in unsafe operation
        nan_count = np.sum(np.isnan(unsafe_log))
        inf_count = np.sum(np.isinf(unsafe_log))
        
        print(f"Unsafe operation: {nan_count} NaNs, {inf_count} infinities")
        print(f"Safe operation: {np.sum(np.isnan(safe_log_result))} NaNs")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()