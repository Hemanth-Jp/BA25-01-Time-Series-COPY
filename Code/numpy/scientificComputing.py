#!/usr/bin/env python3
"""
NumPy Scientific Computing Application Example

This module demonstrates NumPy's capabilities in scientific computing
including signal processing, numerical integration, and statistical analysis.


Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt

def signal_processing_example():
    """
    Demonstrate signal processing operations using NumPy.
    
    Returns:
        dict: Signal processing results
    """
    print("=== Signal Processing Example ===")
    
    # Generate a composite signal
    t = np.linspace(0, 1, 1000)
    frequency1, frequency2 = 5, 20
    signal = (np.sin(2 * np.pi * frequency1 * t) + 
              0.5 * np.sin(2 * np.pi * frequency2 * t) +
              0.1 * np.random.normal(size=len(t)))
    
    # Apply FFT for frequency analysis
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # Filter the signal (simple low-pass filter)
    cutoff_freq = 15
    fft_filtered = fft_result.copy()
    fft_filtered[np.abs(frequencies) > cutoff_freq] = 0
    filtered_signal = np.fft.ifft(fft_filtered).real
    
    # Calculate signal statistics
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    signal_rms = np.sqrt(np.mean(signal**2))
    
    print(f"Signal length: {len(signal)}")
    print(f"Signal mean: {signal_mean:.4f}")
    print(f"Signal std: {signal_std:.4f}")
    print(f"Signal RMS: {signal_rms:.4f}")
    
    return {
        'time': t, 'signal': signal, 'filtered': filtered_signal,
        'fft': fft_result, 'frequencies': frequencies
    }

def numerical_integration_example():
    """
    Demonstrate numerical integration techniques.
    
    Returns:
        dict: Integration results
    """
    print("\n=== Numerical Integration Example ===")
    
    # Define a function to integrate: f(x) = x^2 * sin(x)
    def integrand(x):
        return x**2 * np.sin(x)
    
    # Integration bounds
    a, b = 0, np.pi
    
    # Trapezoidal rule
    n_points = 1000
    x = np.linspace(a, b, n_points)
    y = integrand(x)
    trap_result = np.trapz(y, x)
    
    # Simpson's rule (manual implementation)
    def simpsons_rule(func, a, b, n):
        if n % 2 == 1:
            n += 1  # Ensure even number of intervals
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        return h/3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    
    simpson_result = simpsons_rule(integrand, a, b, 1000)
    
    # Monte Carlo integration
    n_samples = 100000
    random_x = np.random.uniform(a, b, n_samples)
    monte_carlo_result = (b - a) * np.mean(integrand(random_x))
    
    print(f"Trapezoidal rule: {trap_result:.6f}")
    print(f"Simpson's rule: {simpson_result:.6f}")
    print(f"Monte Carlo: {monte_carlo_result:.6f}")
    
    return {
        'function': integrand, 'x': x, 'y': y,
        'trapezoidal': trap_result, 'simpson': simpson_result,
        'monte_carlo': monte_carlo_result
    }

def statistical_analysis_example():
    """
    Perform statistical analysis on generated data.
    
    Returns:
        dict: Statistical analysis results
    """
    print("\n=== Statistical Analysis Example ===")
    
    # Generate sample datasets
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    exponential_data = np.random.exponential(2, 1000)
    
    # Descriptive statistics
    def compute_stats(data, name):
        stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75)
        }
        
        print(f"{name} Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        return stats
    
    normal_stats = compute_stats(normal_data, "Normal Distribution")
    exp_stats = compute_stats(exponential_data, "Exponential Distribution")
    
    # Correlation analysis
    combined_data = np.column_stack([normal_data, exponential_data])
    correlation_matrix = np.corrcoef(combined_data.T)
    
    print(f"Correlation matrix:\n{correlation_matrix}")
    
    return {
        'normal_data': normal_data, 'exp_data': exponential_data,
        'normal_stats': normal_stats, 'exp_stats': exp_stats,
        'correlation': correlation_matrix
    }

if __name__ == "__main__":
    # Run scientific computing demonstrations
    signal_results = signal_processing_example()
    integration_results = numerical_integration_example()
    stats_results = statistical_analysis_example()
    
    print("\n=== Advanced Application: Image Processing ===")
    # Create a synthetic image
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    image = np.exp(-(X**2 + Y**2))
    
    # Add noise
    noisy_image = image + 0.1 * np.random.normal(size=image.shape)
    
    # Apply 2D convolution for smoothing
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    from scipy import ndimage
    smoothed_image = ndimage.convolve(noisy_image, kernel)
    
    print(f"Original image shape: {image.shape}")
    print(f"Image value range: [{np.min(image):.3f}, {np.max(image):.3f}]")
    print(f"Noise level (std): {np.std(noisy_image - image):.4f}")
