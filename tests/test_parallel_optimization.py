"""
Test for parallel optimization using multiprocessing
"""

import json
import time
import pickle
import multiprocessing
from pathlib import Path
from app import MatrixApp
from app_parallel import MatrixAppParallel
from data_models import CalculationParams, ProcessingMode


REPRESENTATIVE_CASES = [
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_0_1.pickle",
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_1_2.pickle",
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_0_2.pickle",
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_1_1.pickle",
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_0_1.pickle",
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_1_2.pickle",
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_0_2.pickle",
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_1_1.pickle",
]


def _test_parallel_configuration(test_case_path, num_workers, enable_pruning):
    """Test a specific parallel configuration"""
    with open(test_case_path, "rb") as f:
        test_case = pickle.load(f, fix_imports=False)
    
    matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
    
    # Create params with parallel configuration
    params = CalculationParams(
        n=test_case.params.n,
        mode=test_case.params.mode,
        relaxation=test_case.params.relaxation,
        top_k=test_case.params.top_k,
        enable_pruning=enable_pruning,  # Test parallelization without pruning first
        num_workers=num_workers
    )
    
    # Warm up
    res, stats = MatrixAppParallel.max_n_subset_intersection_brute_force(
        matrix, params, {}
    )
    
    # Time the execution
    times = []
    for _ in range(3):
        start = time.perf_counter()
        res, stats = MatrixAppParallel.max_n_subset_intersection_brute_force(
            matrix, params, {}
        )
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    
    # Verify correctness
    correctness = all([
        result == desired for result, desired in zip(res, test_case.expected_results)
    ])
    
    return avg_time, correctness, stats


def test_parallel_configuration():
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"System has {num_cores} CPU cores available")
    
    # Test different worker configurations
    worker_configs = [1, 2, min(8, num_cores)]
    worker_configs = list(set(worker_configs))  # Remove duplicates
    worker_configs.sort()
    
    results_summary = {}
    
    print("\n" + "=" * 80)
    print("TESTING PARALLEL OPTIMIZATION")
    print("=" * 80)

    enable_pruning = True

    if enable_pruning:
        print("Testing with pruning")
        results_path = "tests/parallel_optimization_results_pruning.json"
    else:
        print("Testing without pruning")
        results_path = "tests/parallel_optimization_results_no_pruning.json"
    
    for test_case_path in REPRESENTATIVE_CASES:
        case_name = Path(test_case_path).stem
        print(f"\n{case_name}:")
        
        with open(test_case_path, "rb") as f:
            test_case = pickle.load(f, fix_imports=False)
        print(f"  n={test_case.params.n}, relaxation={test_case.params.relaxation}, top_k={test_case.params.top_k}")
        
        case_results = {}
        
        for num_workers in worker_configs:
            avg_time, correctness, stats = _test_parallel_configuration(test_case_path, num_workers, enable_pruning)
            
            case_results[num_workers] = {
                "time": avg_time,
                "correctness": correctness,
                "stats": stats
            }
            
            print(f"  Workers={num_workers}: {avg_time:.4f}s, Correctness: {'✓' if correctness else '✗'}")
        
        # Calculate speedups relative to single-threaded
        baseline_time = case_results[1]["time"]
        for num_workers in worker_configs:
            case_results[num_workers]["speedup"] = baseline_time / case_results[num_workers]["time"]
        
        results_summary[case_name] = case_results
        
        # Show speedups
        print("  Speedups:")
        for num_workers in worker_configs[1:]:
            speedup = case_results[num_workers]["speedup"]
            print(f"    {num_workers} workers: {speedup:.2f}x")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("PARALLEL OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    # Calculate average speedups for each worker configuration
    for num_workers in worker_configs[1:]:
        speedups = [results_summary[case][num_workers]["speedup"] 
                   for case in results_summary]
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        
        print(f"\n{num_workers} Workers:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Min speedup: {min_speedup:.2f}x")
        print(f"  Max speedup: {max_speedup:.2f}x")
    
    # Test with pruning + parallelization
    print("\n" + "=" * 80)
    print("TESTING COMBINED PRUNING + PARALLEL OPTIMIZATION")
    print("=" * 80)
    
    combined_results = {}
    best_num_workers = 4  # Use 4 workers as a good default
    
    for test_case_path in REPRESENTATIVE_CASES:
        case_name = Path(test_case_path).stem
        print(f"\n{case_name}:")
        
        with open(test_case_path, "rb") as f:
            test_case = pickle.load(f, fix_imports=False)
        
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        
        # Test with pruning + parallelization
        params = CalculationParams(
            n=test_case.params.n,
            mode=test_case.params.mode,
            relaxation=test_case.params.relaxation,
            top_k=test_case.params.top_k,
            enable_pruning=True,
            num_workers=best_num_workers
        )
        
        # Warm up
        MatrixAppParallel.max_n_subset_intersection_brute_force(matrix, params, {})
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            res, stats = MatrixAppParallel.max_n_subset_intersection_brute_force(
                matrix, params, {}
            )
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        
        # Compare to baseline (no optimization)
        baseline_time = results_summary[case_name][1]["time"]
        speedup = baseline_time / avg_time
        
        combined_results[case_name] = {
            "baseline_time": baseline_time,
            "combined_time": avg_time,
            "speedup": speedup,
            "pruning_rate": stats.get("pruning_rate", 0) * 100 if stats else 0
        }
        
        print(f"  Baseline: {baseline_time:.4f}s")
        print(f"  Pruning + {best_num_workers} workers: {avg_time:.4f}s")
        print(f"  Combined speedup: {speedup:.2f}x")
        if stats and "pruning_rate" in stats:
            print(f"  Pruning rate: {stats['pruning_rate']*100:.1f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMBINED OPTIMIZATION SUMMARY (Pruning + Parallel)")
    print("=" * 80)
    
    speedups = [v["speedup"] for v in combined_results.values()]
    print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")
    print(f"Min speedup: {min(speedups):.2f}x")
    print(f"Max speedup: {max(speedups):.2f}x")
    
    # Save all results
    with open(results_path, "w") as f:
        # Convert to JSON-serializable format
        json_results = {
            "parallel_only": {},
            "combined": {}
        }
        
        for case in results_summary:
            json_results["parallel_only"][case] = {}
            for num_workers in worker_configs:
                json_results["parallel_only"][case][str(num_workers)] = {
                    "time": results_summary[case][num_workers]["time"],
                    "speedup": results_summary[case][num_workers].get("speedup", 1.0),
                    "correctness": results_summary[case][num_workers]["correctness"]
                }
        
        for case in combined_results:
            json_results["combined"][case] = combined_results[case]
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")