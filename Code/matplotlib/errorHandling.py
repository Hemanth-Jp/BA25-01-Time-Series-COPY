"""
Comprehensive Error Handling with Matplotlib

This module demonstrates robust error handling patterns for Matplotlib applications:
- Data validation and preprocessing
- Backend configuration issues
- Memory management
- Graceful degradation strategies


Version: 1.0
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatplotlibErrorHandler:
    """
    Comprehensive error handling for Matplotlib operations.
    
    Provides robust methods for:
    - Backend management
    - Data validation
    - Memory management
    - Error recovery
    """
    
    def __init__(self):
        """Initialize error handler with safe defaults."""
        self.original_backend = matplotlib.get_backend()
        self.figure_count = 0
        self.max_figures = 10
        
    def safe_backend_setup(self, preferred_backend='Agg'):
        """
        Safely configure matplotlib backend with fallback options.
        
        Args:
            preferred_backend (str): Preferred backend name
            
        Returns:
            bool: True if backend setup successful, False otherwise
        """
        try:
            # List of fallback backends in order of preference
            fallback_backends = [preferred_backend, 'Agg', 'TkAgg', 'Qt5Agg']
            
            for backend in fallback_backends:
                try:
                    matplotlib.use(backend, force=True)
                    logger.info(f"Successfully set backend to: {backend}")
                    return True
                except ImportError as e:
                    logger.warning(f"Backend {backend} not available: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error setting backend {backend}: {e}")
                    continue
            
            logger.error("No suitable backend found!")
            return False
            
        except Exception as e:
            logger.error(f"Critical error in backend setup: {e}")
            return False
    
    def validate_data(self, *arrays):
        """
        Validate input data for plotting.
        
        Args:
            *arrays: Variable number of data arrays to validate
            
        Returns:
            tuple: Validated and cleaned data arrays
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            validated_arrays = []
            
            for i, arr in enumerate(arrays):
                # Convert to numpy array
                try:
                    data = np.asarray(arr, dtype=float)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Array {i} contains invalid data: {e}")
                
                # Check for empty arrays
                if data.size == 0:
                    raise ValueError(f"Array {i} is empty")
                
                # Handle infinite and NaN values
                if np.any(np.isinf(data)):
                    logger.warning(f"Array {i} contains infinite values, replacing with NaN")
                    data = np.where(np.isinf(data), np.nan, data)
                
                if np.any(np.isnan(data)):
                    logger.warning(f"Array {i} contains NaN values")
                    # Option 1: Remove NaN values
                    # data = data[~np.isnan(data)]
                    # Option 2: Replace with interpolation or mean
                    if np.all(np.isnan(data)):
                        raise ValueError(f"Array {i} contains only NaN values")
                
                validated_arrays.append(data)
            
            # Check array compatibility
            if len(validated_arrays) > 1:
                lengths = [len(arr) for arr in validated_arrays]
                if len(set(lengths)) > 1:
                    min_length = min(lengths)
                    logger.warning(f"Arrays have different lengths, truncating to {min_length}")
                    validated_arrays = [arr[:min_length] for arr in validated_arrays]
            
            return tuple(validated_arrays)
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def safe_plot_creation(self, plot_func, *args, **kwargs):
        """
        Safely create plots with comprehensive error handling.
        
        Args:
            plot_func: Plotting function to execute
            *args: Arguments for plotting function
            **kwargs: Keyword arguments for plotting function
            
        Returns:
            tuple: (figure, axes) or (None, None) on failure
        """
        fig, ax = None, None
        
        try:
            # Check figure limit
            if self.figure_count >= self.max_figures:
                logger.warning("Maximum figure limit reached, cleaning up old figures")
                plt.close('all')
                self.figure_count = 0
            
            # Create figure with error handling
            try:
                fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))
                self.figure_count += 1
            except Exception as e:
                logger.error(f"Failed to create figure: {e}")
                return None, None
            
            # Execute plotting function
            result = plot_func(ax, *args, **kwargs)
            
            # Basic plot validation
            if not ax.lines and not ax.collections and not ax.patches:
                logger.warning("Plot appears to be empty")
            
            return fig, ax
            
        except MemoryError:
            logger.error("Memory error during plotting - try reducing data size")
            if fig:
                plt.close(fig)
            return None, None
            
        except Exception as e:
            logger.error(f"Error during plot creation: {e}")
            if fig:
                plt.close(fig)
            return None, None
    
    def memory_efficient_plotting(self, x, y, chunk_size=10000):
        """
        Plot large datasets efficiently by chunking data.
        
        Args:
            x, y: Data arrays
            chunk_size (int): Size of data chunks
            
        Returns:
            tuple: (figure, axes) or (None, None) on failure
        """
        try:
            # Validate data
            x, y = self.validate_data(x, y)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot data in chunks
            for i in range(0, len(x), chunk_size):
                end_idx = min(i + chunk_size, len(x))
                x_chunk = x[i:end_idx]
                y_chunk = y[i:end_idx]
                
                # Plot chunk
                ax.plot(x_chunk, y_chunk, 'b-', linewidth=1, alpha=0.7)
            
            ax.set_xlabel('X values')
            ax.set_ylabel('Y values')
            ax.set_title(f'Large Dataset Plot ({len(x)} points)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            logger.info(f"Successfully plotted {len(x)} points in chunks")
            return fig, ax
            
        except Exception as e:
            logger.error(f"Error in memory-efficient plotting: {e}")
            return None, None

def demonstrate_error_handling():
    """
    Demonstrate various error handling scenarios.
    """
    handler = MatplotlibErrorHandler()
    
    print("=== Matplotlib Error Handling Demonstration ===\n")
    
    # 1. Backend setup
    print("1. Testing backend setup...")
    success = handler.safe_backend_setup('Agg')
    print(f"Backend setup: {'Success' if success else 'Failed'}\n")
    
    # 2. Data validation
    print("2. Testing data validation...")
    try:
        # Valid data
        x_valid = np.linspace(0, 10, 100)
        y_valid = np.sin(x_valid)
        x_clean, y_clean = handler.validate_data(x_valid, y_valid)
        print(" Valid data passed validation")
        
        # Invalid data with NaN
        y_invalid = y_valid.copy()
        y_invalid[50:60] = np.nan
        x_clean, y_clean = handler.validate_data(x_valid, y_invalid)
        print(" Data with NaN values handled")
        
        # Mismatched lengths
        x_short = x_valid[:50]
        x_clean, y_clean = handler.validate_data(x_short, y_valid)
        print(" Mismatched array lengths handled")
        
    except Exception as e:
        print(f" Data validation error: {e}")
    
    print()
    
    # 3. Safe plotting
    print("3. Testing safe plot creation...")
    
    def simple_plot(ax, x, y):
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title('Safe Plot Creation Test')
        return ax
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    fig, ax = handler.safe_plot_creation(simple_plot, x, y)
    if fig and ax:
        print(" Safe plot creation successful")
        plt.close(fig)
    else:
        print(" Safe plot creation failed")
    
    # 4. Memory-efficient plotting
    print("4. Testing memory-efficient plotting...")
    
    # Generate large dataset
    large_x = np.linspace(0, 100, 50000)
    large_y = np.sin(large_x) + 0.1 * np.random.randn(len(large_x))
    
    fig, ax = handler.memory_efficient_plotting(large_x, large_y, chunk_size=5000)
    if fig and ax:
        print(f" Memory-efficient plotting successful for {len(large_x)} points")
        plt.close(fig)
    else:
        print(" Memory-efficient plotting failed")
    
    print("\n=== Error Handling Demonstration Complete ===")

def safe_export_figure(fig, filename, formats=['png'], dpi=300, **kwargs):
    """
    Safely export figure to multiple formats with error handling.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        formats (list): List of formats to export
        dpi (int): Resolution for raster formats
        **kwargs: Additional arguments for savefig
    """
    if fig is None:
        logger.error("Cannot export: figure is None")
        return False
    
    success_count = 0
    
    for fmt in formats:
        try:
            full_filename = f"{filename}.{fmt}"
            
            # Format-specific settings
            save_kwargs = kwargs.copy()
            if fmt in ['png', 'jpg', 'jpeg']:
                save_kwargs['dpi'] = dpi
            elif fmt in ['pdf', 'svg']:
                save_kwargs.pop('dpi', None)  # Vector formats don't use dpi
            
            # Save figure
            fig.savefig(full_filename, format=fmt, bbox_inches='tight', **save_kwargs)
            logger.info(f"Successfully exported to {full_filename}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to export to {fmt}: {e}")
    
    return success_count > 0

if __name__ == "__main__":
    print("Running Matplotlib Error Handling Examples...")
    
    try:
        demonstrate_error_handling()
        print("Error handling demonstration completed successfully!")
        
    except Exception as e:
        print(f"Critical error in demonstration: {e}")
        import traceback
        traceback.print_exc()