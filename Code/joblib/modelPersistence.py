"""
Model Persistence and Advanced Caching with Joblib

This module demonstrates Joblib's model persistence capabilities and advanced
caching strategies for machine learning workflows. It showcases model
serialization, cache integration, and performance optimization.

"""

import joblib
import numpy as np
import time
import os
import shutil
import tempfile
from joblib import Memory
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_sample_dataset(n_samples=1000, n_features=20):
    """
    Create a sample classification dataset.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        
    Returns:
        tuple: X, y arrays for training
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        random_state=42
    )
    return X, y


def demonstrate_model_persistence():
    """Demonstrate basic model persistence with Joblib."""
    print("=== Model Persistence Demo ===")
    
    # Create and train a model
    X, y = create_sample_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        model_filename = tmp_file.name
    
    print(f"Saving model to {model_filename}")
    joblib.dump(model, model_filename)
    
    print("Loading model from disk...")
    loaded_model = joblib.load(model_filename)
    loaded_pred = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, loaded_pred)
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
    
    os.remove(model_filename)
    
    return model, X_test, y_test


def demonstrate_cached_model_training():
    """Demonstrate caching in model training pipelines."""
    print("\n=== Cached Model Training Demo ===")
    
    # Use a writable temporary cache directory
    cache_dir = tempfile.mkdtemp()
    memory = Memory(location=cache_dir, verbose=1)
    
    @memory.cache
    def train_cached_model(n_estimators, max_depth, n_samples):
        """
        Train a model with caching to avoid retraining.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            n_samples (int): Dataset size
            
        Returns:
            tuple: Trained model and accuracy
        """
        print(f"Training model: trees={n_estimators}, depth={max_depth}, samples={n_samples}")
        X, y = create_sample_dataset(n_samples=n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

    configs = [
        (50, 5, 1000),
        (100, 10, 1000),
        (50, 5, 1000),  # This should hit the cache
    ]
    
    for n_est, depth, samples in configs:
        start_time = time.time()
        model, acc = train_cached_model(n_est, depth, samples)
        training_time = time.time() - start_time
        print(f"Config ({n_est}, {depth}, {samples}): "
              f"accuracy={acc:.4f}, time={training_time:.3f}s")
    
    shutil.rmtree(cache_dir)


def demonstrate_compression_options():
    """Demonstrate different compression options for model storage."""
    print("\n=== Compression Options Demo ===")
    
    X, y = create_sample_dataset(n_samples=2000)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    compression_methods = [
        ('no_compression', 0),
        ('zlib_fast', 1),
        ('zlib_medium', 6),
        ('zlib_best', 9)
    ]
    
    for comp_name, comp_level in compression_methods:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            filename = tmp_file.name
        
        start_time = time.time()
        joblib.dump(model, filename, compress=comp_level)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize(filename) / 1024  # KB
        
        start_time = time.time()
        loaded_model = joblib.load(filename)
        load_time = time.time() - start_time
        
        print(f"{comp_name}: size={file_size:.1f}KB, "
              f"save={save_time:.3f}s, load={load_time:.3f}s")
        
        os.remove(filename)


def main():
    """Main demonstration function."""
    print("Joblib Model Persistence and Caching Demonstration")
    
    try:
        demonstrate_model_persistence()
        demonstrate_cached_model_training()
        demonstrate_compression_options()
        print("\n=== Model Persistence Demo Complete ===")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()