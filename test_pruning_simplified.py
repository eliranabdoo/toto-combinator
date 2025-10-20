"""
Simplified test for pruning optimization using the enable_pruning flag
"""

import json
import time
import pickle
from pathlib import Path
from app_testing import MatrixApp
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


def main():
    results_summary = {
        "baseline": {},
        "pruning": {},
        "speedups": {}
    }
    
    print("=" * 80)
    print("TESTING PRUNING OPTIMIZATION WITH FLAG APPROACH")
    print("=" * 80)
    
    for test_case_path in REPRESENTATIVE_CASES:
        with open(test_case_path, "rb") as f:
            test_case = pickle.load(f, fix_imports=False)
        
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        case_name = Path(test_case_path).stem
        
        print(f"\n{case_name}:")
        print(f"  n={test_case.params.n}, relaxation={test_case.params.relaxation}, top_k={test_case.params.top_k}")
        
        # Test WITHOUT pruning (baseline)
        params_no_pruning = CalculationParams(
            n=test_case.params.n,
            mode=test_case.params.mode,
            relaxation=test_case.params.relaxation,
            top_k=test_case.params.top_k,
            enable_pruning=False
        )
        
        # Warm up
        MatrixApp.run_calculation(matrix=matrix, params=params_no_pruning, return_stats=True)
        
        # Time baseline
        times_baseline = []
        for _ in range(3):
            start = time.perf_counter()
            res_baseline, stats_baseline = MatrixApp.run_calculation(
                matrix=matrix, params=params_no_pruning, return_stats=True
            )
            end = time.perf_counter()
            times_baseline.append(end - start)
        
        avg_baseline = sum(times_baseline) / len(times_baseline)
        
        # Test WITH pruning  
        params_with_pruning = CalculationParams(
            n=test_case.params.n,
            mode=test_case.params.mode,
            relaxation=test_case.params.relaxation,
            top_k=test_case.params.top_k,
            enable_pruning=True
        )
        
        # Warm up
        MatrixApp.run_calculation(matrix=matrix, params=params_with_pruning, return_stats=True)
        
        # Time pruning
        times_pruning = []
        for _ in range(3):
            start = time.perf_counter()
            res_pruning, stats_pruning = MatrixApp.run_calculation(
                matrix=matrix, params=params_with_pruning, return_stats=True
            )
            end = time.perf_counter()
            times_pruning.append(end - start)
        
        avg_pruning = sum(times_pruning) / len(times_pruning)
        
        # Verify correctness
        correctness = all([
            result == desired for result, desired in zip(res_pruning, test_case.expected_results)
        ])
        
        # Calculate speedup
        speedup = avg_baseline / avg_pruning
        
        # Store results
        results_summary["baseline"][case_name] = avg_baseline
        results_summary["pruning"][case_name] = avg_pruning
        results_summary["speedups"][case_name] = speedup
        
        # Print results
        print(f"  Baseline time: {avg_baseline:.4f}s")
        print(f"  Pruning time: {avg_pruning:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Correctness: {'✓ PASS' if correctness else '✗ FAIL'}")
        
        if stats_pruning:
            pruning_rate = stats_pruning.get('pruning_rate', 0) * 100
            print(f"  Pruning rate: {pruning_rate:.1f}%")
            print(f"  Pruned choices: {stats_pruning.get('pruned_choices', 0):,} / {stats_pruning.get('total_choices', 0):,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PRUNING OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    all_speedups = list(results_summary["speedups"].values())
    avg_speedup = sum(all_speedups) / len(all_speedups)
    min_speedup = min(all_speedups)
    max_speedup = max(all_speedups)
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Min speedup: {min_speedup:.2f}x")  
    print(f"Max speedup: {max_speedup:.2f}x")
    
    # Save results
    with open("pruning_results_simplified.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nResults saved to pruning_results_simplified.json")


if __name__ == "__main__":
    main()