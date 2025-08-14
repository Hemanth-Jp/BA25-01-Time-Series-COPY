"""
Scientific Plotting with Matplotlib

This module demonstrates specialized plotting techniques for scientific applications:
- Contour plots and vector fields
- 3D surface visualizations
- Polar plots and log scales
- Mathematical function plotting


Version: 1.0
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def create_contour_plot():
    """
    Create contour plots for mathematical functions.
    
    Demonstrates:
    - 2D contour plotting
    - Filled contours
    - Vector field overlay
    - Scientific color schemes
    """
    # Generate mesh grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Define mathematical function
    Z = np.exp(-(X**2 + Y**2)) * np.cos(2*np.pi*np.sqrt(X**2 + Y**2))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Contour plot
    contour = ax1.contour(X, Y, Z, levels=15, colors='black', linewidths=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)
    filled_contour = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(filled_contour, ax=ax1, label='Function Value')
    ax1.set_title('Contour Plot: f(x,y) = exp(-(x^2+y^2)) * cos(2*pi*sqrt(x^2+y^2))', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    
    # Vector field
    x_vec = np.linspace(-3, 3, 20)
    y_vec = np.linspace(-3, 3, 20)
    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
    
    # Calculate gradient
    dx = -2*X_vec * np.exp(-(X_vec**2 + Y_vec**2))
    dy = -2*Y_vec * np.exp(-(X_vec**2 + Y_vec**2))
    
    ax2.quiver(X_vec, Y_vec, dx, dy, alpha=0.7)
    ax2.contour(X, Y, Z, levels=10, colors='red', linewidths=1)
    ax2.set_title('Vector Field with Contour Lines', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_3d_surface():
    """
    Create 3D surface plots for scientific visualization.
    
    Demonstrates:
    - 3D surface plotting
    - Wireframe plots
    - Custom viewing angles
    - Surface coloring
    """
    # Generate 3D data
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2 + 0.1)
    
    # Create 3D subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9)
    ax1.set_title('3D Surface Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Wireframe plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_wireframe(X, Y, Z, color='blue', alpha=0.7)
    ax2.set_title('3D Wireframe Plot', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Contour 3D
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.contour3D(X, Y, Z, 20, cmap='viridis')
    ax3.set_title('3D Contour Plot', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def create_polar_and_log_plots():
    """
    Create specialized polar and logarithmic plots.
    
    Demonstrates:
    - Polar coordinate plotting
    - Logarithmic scales
    - Scientific data representation
    - Angular measurements
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 10))
    
    # Polar plot
    ax1 = plt.subplot(221, projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2 + np.cos(3*theta)
    ax1.plot(theta, r, 'b-', linewidth=2)
    ax1.set_title('Polar Plot: Rose Curve', pad=20, fontsize=12, fontweight='bold')
    ax1.grid(True)
    
    # Log-log plot
    ax2 = plt.subplot(222)
    x_log = np.logspace(0, 3, 50)
    y_log = x_log**2.5
    ax2.loglog(x_log, y_log, 'ro-', markersize=4)
    ax2.set_xlabel('X (log scale)')
    ax2.set_ylabel('Y (log scale)')
    ax2.set_title('Log-Log Plot: Power Law', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    
    # Semi-log plot
    ax3 = plt.subplot(223)
    x_semi = np.linspace(0, 10, 100)
    y_semi = np.exp(x_semi/2)
    ax3.semilogy(x_semi, y_semi, 'g-', linewidth=2)
    ax3.set_xlabel('X (linear scale)')
    ax3.set_ylabel('Y (log scale)')
    ax3.set_title('Semi-Log Plot: Exponential Growth', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Error function plot
    ax4 = plt.subplot(224)
    x_erf = np.linspace(-3, 3, 100)
    y_erf = 2/np.sqrt(np.pi) * np.cumsum(np.exp(-x_erf**2)) * (x_erf[1] - x_erf[0])
    ax4.plot(x_erf, y_erf, 'purple', linewidth=2, label='Error Function')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('X')
    ax4.set_ylabel('erf(X)')
    ax4.set_title('Error Function Approximation', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Scientific Plotting Examples...")
    
    try:
        create_contour_plot()
        create_3d_surface()
        create_polar_and_log_plots()
        print("Scientific plotting examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()