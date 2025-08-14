"""
Comprehensive Error Handling with Joblib

This module demonstrates robust error handling patterns for Joblib applications,
including cache corruption handling, memory management, serialization errors,
and parallel processing failure recovery.


"""

import joblib
import numpy as np
import os
import shutil
import time
from joblib import Memory, Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import warnings
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustCache:
    """
    A robust caching wrapper that handles common cache-related errors.
    """
    
    def __init__(self, cache_dir='./robust_cache', max_retries=3):
        """
        Initialize robust cache with error handling.
        
        Args:
            cache_dir (str): Cache directory path
            max_retries (int): Maximum retry attempts
        """
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.memory = None
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup cache with error handling."""
        try:
            self.memory = Memory(location=self.cache_dir, verbose=1)
            logger.info(f"Cache initialized at: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            self._handle_cache_setup_error()
    
    def _handle_cache_setup_error(self):
        """Handle cache setup errors by creating backup location."""
        try:
            backup_dir = f"{self.cache_dir}_backup_{int(time.time())}"
            self.memory = Memory(location=backup_dir, verbose=1)
            logger.warning(f"Using backup cache location: {backup_dir}")
        except Exception as e:
            logger.error(f"Backup cache setup failed: {e}")
            # Fall back to no caching
            self.memory = Memory(location=None)
            logger.warning("Caching disabled due to setup errors")
    
    def cache_function(self, func):
        """
        Cache a function with error handling.
        
        Args:
            func: Function to cache
            
        Returns:
            Cached function with error handling
        """
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    cached_func = self.memory.cache(func)
                    return cached_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Cache attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        logger.error("All cache attempts failed, executing without cache")
                        return func(*args, **kwargs)
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
        return wrapper


def problematic_function(data, should_fail=False):
    """
    A function that may fail in various ways.
    
    Args:
        data: Input data
        should_fail (bool): Whether to simulate failure
        
    Returns:
        Processed data
        
    Raises:
        ValueError: When should_fail is True
        MemoryError: When data is too large
    """
    if should_fail:
        raise ValueError("Simulated processing error")
    
    if hasattr(data, '__len__') and len(data) > 1000000:
        raise MemoryError("Data too large for processing")
    
    return np.mean(data) if hasattr(data, '__len__') else data * 2


def demonstrate_cache_error_handling():
    """Demonstrate handling of cache-related errors."""
    print("=== Cache Error Handling Demo ===")
    
    robust_cache = RobustCache()
    
    # Test normal operation
    print("Testing normal cache operation:")
    cached_func = robust_cache.cache_function(problematic_function)
    
    test_data = np.random.randn(1000)
    result = cached_func(test_data)
    print(f"Normal operation result: {result:.4f}")
    
    # Test with corrupted cache directory
    print("\nTesting with cache corruption:")
    try:
        # Simulate cache corruption by creating invalid files
        cache_path = os.path.join(robust_cache.cache_dir, 'corrupted_file')
        if os.path.exists(robust_cache.cache_dir):
            with open(cache_path, 'w') as f:
                f.write("corrupted cache data")
        
        result = cached_func(test_data)
        print(f"Recovered from corruption: {result:.4f}")
        
    except Exception as e:
        print(f"Cache error handling failed: {e}")


def safe_parallel_execution(func, arguments, max_workers=None, fallback_sequential=True):
    """
    Execute function in parallel with comprehensive error handling.
    
    Args:
        func: Function to execute
        arguments: List of argument tuples
        max_workers (int): Maximum number of workers
        fallback_sequential (bool): Whether to fall back to sequential execution
        
    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = min(4, len(arguments))
    
    backends_to_try = ['loky', 'threading', 'sequential']
    
    for backend in backends_to_try:
        try:
            if backend == 'sequential':
                if not fallback_sequential:
                    raise Exception("Sequential fallback disabled")
                logger.info("Falling back to sequential execution")
                return [func(*args) if isinstance(args, tuple) else func(args) 
                       for args in arguments]
            
            logger.info(f"Trying parallel execution with {backend} backend")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                results = Parallel(
                    n_jobs=max_workers,
                    backend=backend,
                    timeout=30
                )(
                    delayed(func)(*args) if isinstance(args, tuple) else delayed(func)(args)
                    for args in arguments
                )
                
                logger.info(f"Parallel execution successful with {backend}")
                return results
                
        except Exception as e:
            logger.warning(f"Backend {backend} failed: {e}")
            if backend != 'sequential':
                continue
            else:
                logger.error("All execution methods failed")
                raise
    
    return []


def demonstrate_parallel_error_handling():
    """Demonstrate error handling in parallel processing."""
    print("\n=== Parallel Error Handling Demo ===")
    
    # Test data with some problematic cases
    test_cases = [
        (np.random.randn(100), False),
        (np.random.randn(200), False),
        (np.random.randn(150), True),   # This will fail
        (np.random.randn(300), False),
        (np.random.randn(50), False),
    ]
    
    print("Testing parallel execution with mixed success/failure:")
    
    def safe_wrapper(data, should_fail):
        """Wrapper that handles individual task failures."""
        try:
            return problematic_function(data, should_fail)
        except Exception as e:
            logger.warning(f"Task failed: {e}")
            return None  # Return None for failed tasks
    
    results = safe_parallel_execution(safe_wrapper, test_cases, max_workers=2)
    
    print(f"Results: {len([r for r in results if r is not None])} successful, "
          f"{len([r for r in results if r is None])} failed")
    
    # Clean up any remaining processes
    try:
        executor = get_reusable_executor(max_workers=2)
        executor.shutdown(wait=True)
    except:
        pass


def demonstrate_serialization_error_handling():
    """Demonstrate handling of serialization errors."""
    print("\n=== Serialization Error Handling Demo ===")
    
    class UnserializableClass:
        """A class that cannot be pickled."""
        def __init__(self):
            self.data = lambda x: x * 2  # Lambda functions can't be pickled
    
    def safe_dump(obj, filename, fallback_method='json'):
        """
        Safely dump objects with fallback methods.
        
        Args:
            obj: Object to serialize
            filename (str): Output filename
            fallback_method (str): Fallback serialization method
            
        Returns:
            bool: Success status
        """
        try:
            # Try joblib dump first
            joblib.dump(obj, filename)
            logger.info(f"Successfully saved with joblib: {filename}")
            return True
            
        except Exception as e:
            logger.warning(f"Joblib dump failed: {e}")
            
            try:
                if fallback_method == 'json' and hasattr(obj, '__dict__'):
                    import json
                    with open(filename.replace('.pkl', '.json'), 'w') as f:
                        json.dump(obj.__dict__, f)
                    logger.info(f"Saved with JSON fallback: {filename}")
                    return True
                    
            except Exception as e2:
                logger.error(f"Fallback serialization failed: {e2}")
                return False
    
    # Test serialization with different object types
    test_objects = [
        (np.array([1, 2, 3, 4, 5]), 'array.pkl'),
        ({'key': 'value', 'numbers': [1, 2, 3]}, 'dict.pkl'),
        (UnserializableClass(), 'unserializable.pkl'),
    ]
    
    for obj, filename in test_objects:
        success = safe_dump(obj, filename)
        if success and os.path.exists(filename):
            os.remove(filename)
        elif os.path.exists(filename.replace('.pkl', '.json')):
            os.remove(filename.replace('.pkl', '.json'))


def demonstrate_memory_error_handling():
    """Demonstrate handling of memory-related errors."""
    print("\n=== Memory Error Handling Demo ===")
    
    def memory_aware_processing(data_size, max_chunk_size=10000):
        """
        Process large datasets in chunks to avoid memory errors.
        
        Args:
            data_size (int): Size of data to process
            max_chunk_size (int): Maximum chunk size
            
        Returns:
            float: Processed result
        """
        try:
            # Try processing all at once
            data = np.random.randn(data_size)
            return np.mean(data ** 2)
            
        except MemoryError:
            logger.warning(f"Memory error with size {data_size}, using chunked processing")
            
            # Fall back to chunked processing
            total_sum = 0
            total_count = 0
            
            for start in range(0, data_size, max_chunk_size):
                end = min(start + max_chunk_size, data_size)
                chunk_size = end - start
                
                chunk_data = np.random.randn(chunk_size)
                chunk_result = np.sum(chunk_data ** 2)
                
                total_sum += chunk_result
                total_count += chunk_size
            
            return total_sum / total_count
    
    # Test with progressively larger data sizes
    data_sizes = [1000, 10000, 100000, 1000000]
    
    for size in data_sizes:
        try:
            result = memory_aware_processing(size)
            print(f"Size {size}: result = {result:.6f}")
        except Exception as e:
            print(f"Size {size}: failed with {e}")


def main():
    """Main demonstration function."""
    print("Joblib Comprehensive Error Handling Demonstration")
    
    try:
        # Cache error handling
        demonstrate_cache_error_handling()
        
        # Parallel processing error handling
        demonstrate_parallel_error_handling()
        
        # Serialization error handling
        demonstrate_serialization_error_handling()
        
        # Memory error handling
        demonstrate_memory_error_handling()
        
        print("\n=== Error Handling Demo Complete ===")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_dirs = ['./robust_cache', './robust_cache_backup*']
        for pattern in cleanup_dirs:
            import glob
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Cleaned up: {path}")


if __name__ == "__main__":
    main()