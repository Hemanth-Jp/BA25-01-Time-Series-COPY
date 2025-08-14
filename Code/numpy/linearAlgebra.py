#!/usr/bin/env python3
"""
NumPy Linear Algebra Operations Example

This module demonstrates NumPy's linear algebra capabilities including
matrix operations, decompositions, and solving linear systems.


Version: 1.0
"""

import numpy as np
from numpy.linalg import inv, det, eig, svd, solve

def demonstrate_matrix_operations():
    """
    Demonstrate basic matrix operations using NumPy.
    
    Returns:
        dict: Results of matrix operations
    """
    print("=== Matrix Operations Examples ===")
    
    # Create sample matrices
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    
    # Basic operations
    matrix_add = A + B
    matrix_mult = np.dot(A, B)  # Matrix multiplication
    element_mult = A * B        # Element-wise multiplication
    transpose_A = A.T
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A + B:\n{matrix_add}")
    print(f"A @ B (matrix multiplication):\n{matrix_mult}")
    print(f"A * B (element-wise):\n{element_mult}")
    print(f"Transpose of A:\n{transpose_A}")
    
    return {
        'A': A, 'B': B, 'add': matrix_add,
        'mult': matrix_mult, 'transpose': transpose_A
    }

def demonstrate_matrix_decomposition():
    """
    Show matrix decomposition techniques and their applications.
    
    Returns:
        dict: Decomposition results
    """
    print("\n=== Matrix Decomposition Examples ===")
    
    # Create a symmetric positive definite matrix
    A = np.array([[4, 2, 1], [2, 3, 0.5], [1, 0.5, 2]])
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig(A)
    
    # Singular Value Decomposition
    U, s, Vt = svd(A)
    
    # Matrix properties
    determinant = det(A)
    matrix_rank = np.linalg.matrix_rank(A)
    
    print(f"Original Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    print(f"Determinant: {determinant:.4f}")
    print(f"Rank: {matrix_rank}")
    print(f"Singular values: {s}")
    
    return {
        'matrix': A, 'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors, 'svd': (U, s, Vt)
    }

def solve_linear_system():
    """
    Demonstrate solving linear systems of equations.
    
    Returns:
        dict: Solution results
    """
    print("\n=== Linear System Solving ===")
    
    # System: Ax = b
    A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
    b = np.array([1, -2, 0])
    
    # Solve the system
    x = solve(A, b)
    
    # Verify solution
    verification = np.dot(A, x)
    
    print(f"Coefficient matrix A:\n{A}")
    print(f"Constants vector b: {b}")
    print(f"Solution x: {x}")
    print(f"Verification Ax: {verification}")
    print(f"Difference |Ax - b|: {np.linalg.norm(verification - b):.2e}")
    
    return {'A': A, 'b': b, 'solution': x, 'verification': verification}

if __name__ == "__main__":
    # Run demonstrations
    matrix_ops = demonstrate_matrix_operations()
    decompositions = demonstrate_matrix_decomposition()
    linear_system = solve_linear_system()
    
    print("\n=== Advanced Example: Principal Component Analysis ===")
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    data[:, 1] = data[:, 0] + 0.5 * np.random.randn(100)
    data[:, 2] = data[:, 0] - data[:, 1] + 0.3 * np.random.randn(100)
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Eigenvalue decomposition for PCA
    eigenvals, eigenvecs = eig(cov_matrix)
    
    # Sort by eigenvalues
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print(f"Covariance matrix:\n{cov_matrix}")
    print(f"Principal components (eigenvalues): {eigenvals}")
    print(f"Explained variance ratio: {eigenvals / np.sum(eigenvals)}")
