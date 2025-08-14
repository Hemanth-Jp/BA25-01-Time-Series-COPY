"""
Pipeline Integration and Cache Validation with Joblib

This module demonstrates advanced Joblib features including pipeline integration,
cache validation callbacks, and sophisticated workflow management for complex
data science pipelines.

"""

import numpy as np
import pandas as pd
import time
from joblib import Memory, Parallel, delayed
from datetime import datetime, timedelta
import os
import shutil
import tempfile


def simulate_data_source(source_id, n_records=1000):
    print(f"Fetching data from source: {source_id}")
    time.sleep(0.2)

    np.random.seed(hash(source_id) % 2**32)
    data = {
        'id': range(n_records),
        'feature_1': np.random.randn(n_records),
        'feature_2': np.random.exponential(2, n_records),
        'feature_3': np.random.uniform(0, 100, n_records),
        'target': np.random.choice([0, 1], n_records)
    }

    return pd.DataFrame(data)


def process_data_batch(data, processing_type='standard'):
    print(f"Processing {len(data)} records with {processing_type} processing")
    processed = data.copy()

    if processing_type == 'standard':
        processed['feature_1_norm'] = (processed['feature_1'] -
                                       processed['feature_1'].mean()) / processed['feature_1'].std()
        processed['feature_2_log'] = np.log1p(processed['feature_2'])
    elif processing_type == 'advanced':
        processed['feature_interaction'] = (processed['feature_1'] *
                                            processed['feature_2'])
        processed['feature_ratio'] = (processed['feature_3'] /
                                      (processed['feature_2'] + 1))
    return processed


def demonstrate_pipeline_caching():
    print("=== Pipeline Caching Demo ===")
    with tempfile.TemporaryDirectory() as cache_dir:
        memory = Memory(location=cache_dir, verbose=1)

        def cache_validation_callback(metadata):
            duration_valid = metadata.get('duration', 0) > 0.1
            timestamp = metadata.get('timestamp', 0)
            age_valid = time.time() - timestamp < 3600
            return duration_valid and age_valid

        cached_data_source = memory.cache(
            simulate_data_source,
            cache_validation_callback=cache_validation_callback
        )
        cached_processing = memory.cache(
            process_data_batch,
            cache_validation_callback=cache_validation_callback
        )

        sources = ['database_a', 'api_b', 'file_c']
        print("First pipeline run (will cache):")
        pipeline_start = time.time()

        for source in sources:
            raw_data = cached_data_source(source, 500)
            processed_data = cached_processing(raw_data, 'standard')
            print(f"Pipeline step completed for {source}: {len(processed_data)} records")

        first_run_time = time.time() - pipeline_start
        print(f"First run total time: {first_run_time:.3f} seconds")

        print("\nSecond pipeline run (from cache):")
        pipeline_start = time.time()

        for source in sources:
            raw_data = cached_data_source(source, 500)
            processed_data = cached_processing(raw_data, 'standard')

        second_run_time = time.time() - pipeline_start
        print(f"Second run total time: {second_run_time:.3f} seconds")
        print(f"Speed improvement: {first_run_time / second_run_time:.1f}x")


def demonstrate_parallel_pipeline():
    print("\n=== Parallel Pipeline Demo ===")
    data_sources = [f'source_{i}' for i in range(6)]

    print("Sequential processing:")
    start_time = time.time()
    sequential_results = []

    for source in data_sources:
        data = simulate_data_source(source, 300)
        processed = process_data_batch(data, 'advanced')
        sequential_results.append(len(processed))

    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.3f} seconds")

    print("\nParallel processing:")
    start_time = time.time()

    parallel_data = Parallel(n_jobs=3)(
        delayed(simulate_data_source)(source, 300)
        for source in data_sources
    )
    parallel_results = Parallel(n_jobs=3)(
        delayed(process_data_batch)(data, 'advanced')
        for data in parallel_data
    )

    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.3f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.1f}x")
    parallel_lengths = [len(result) for result in parallel_results]
    print(f"Results comparison: {sequential_results == parallel_lengths}")


def demonstrate_cache_management():
    print("\n=== Cache Management Demo ===")
    with tempfile.TemporaryDirectory() as cache_dir:
        memory = Memory(location=cache_dir, verbose=1)

        @memory.cache
        def expensive_analysis(dataset_id, analysis_type):
            print(f"Running {analysis_type} analysis on {dataset_id}")
            time.sleep(0.3)
            data = simulate_data_source(dataset_id, 200)

            if analysis_type == 'correlation':
                return data.corr().values
            elif analysis_type == 'summary':
                return data.describe().values
            else:
                return data.mean().values

        datasets = ['data_1', 'data_2', 'data_3']
        analyses = ['correlation', 'summary', 'mean']

        print("Generating cache entries:")
        for dataset in datasets:
            for analysis in analyses:
                result = expensive_analysis(dataset, analysis)
                print(f"Cached: {dataset} - {analysis}")

        print(f"\nCache location: {memory.location}")
        expensive_analysis.clear()
        print("Cache cleared for expensive_analysis function")


def main():
    print("Joblib Pipeline Integration and Cache Validation Demo")
    try:
        demonstrate_pipeline_caching()
        demonstrate_parallel_pipeline()
        demonstrate_cache_management()
        print("\n=== Pipeline Integration Demo Complete ===")
    except Exception as e:
        print(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()