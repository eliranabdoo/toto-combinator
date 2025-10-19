"""
Optimization Testing Infrastructure for max_n_subset_intersection_brute_force

This module provides the testing framework for validating algorithmic optimizations
while ensuring correctness and measuring performance improvements.
"""

import os
import time
import json
import pickle
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import asdict
from app_testing import MatrixApp
from data_models import MatrixAppTestCase, CalculationParams, CalculationResult


# Representative test cases covering different scenarios
REPRESENTATIVE_CASES = [
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_0_1.pickle",  # Small matrix, n=3, no relaxation
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_1_2.pickle",  # Small matrix, n=3, relaxation=1
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_0_2.pickle",  # Small matrix, n=4, no relaxation
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_1_1.pickle",  # Small matrix, n=4, relaxation=1
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_0_1.pickle",  # Large matrix, n=3, no relaxation
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_1_2.pickle",  # Large matrix, n=3, relaxation=1
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_0_2.pickle",  # Large matrix, n=4, no relaxation
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_1_1.pickle",  # Large matrix, n=4, relaxation=1
]


def load_test_case(test_case_path: str) -> MatrixAppTestCase:
    """Load a test case from pickle file"""
    with open(test_case_path, "rb") as f:
        return pickle.load(f, fix_imports=False)


def collect_baseline_metrics() -> Dict:
    """Collect baseline performance metrics for representative test cases"""
    baseline_results = {}
    
    print("=" * 80)
    print("COLLECTING BASELINE METRICS")
    print("=" * 80)
    
    for test_case_path in REPRESENTATIVE_CASES:
        test_case = load_test_case(test_case_path)
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        
        case_name = Path(test_case_path).stem
        print(f"\nTesting: {case_name}")
        print(f"  Matrix: {test_case.matrix_file_path}")
        print(f"  n={test_case.params.n}, relaxation={test_case.params.relaxation}, top_k={test_case.params.top_k}")
        
        # Warm-up run
        MatrixApp.run_calculation(
            matrix=matrix,
            params=test_case.params,
            return_stats=True
        )
        
        # Actual timing runs
        times = []
        stats_list = []
        for _ in range(3):  # Multiple runs for more stable timing
            start_time = time.perf_counter()
            res, stats = MatrixApp.run_calculation(
                matrix=matrix,
                params=test_case.params,
                return_stats=True
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            stats_list.append(stats)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Use the stats from the first run (they should be identical)
        stats = stats_list[0]
        
        baseline_results[test_case_path] = {
            'case_name': case_name,
            'execution_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'all_times': times,
            'stats': stats,
            'matrix_shape': matrix.shape,
            'params': asdict(test_case.params),
            'num_results': len(res),
            'results': res  # Store for correctness verification
        }
        
        print(f"  Avg execution time: {avg_time:.4f}s")
        print(f"  Min/Max: {min_time:.4f}s / {max_time:.4f}s")
        if stats:
            print(f"  Combinations: {stats.get('combinations', 'N/A')}")
            print(f"  Total choices: {stats.get('total_choices', 'N/A')}")
    
    return baseline_results


def test_optimization(optimization_name: str, optimized_function, baseline_results: Dict) -> Dict:
    """Test a single optimization against representative test cases"""
    results = {
        'optimization_name': optimization_name,
        'correctness': True,
        'performance_gains': {},
        'failures': [],
        'detailed_results': {}
    }
    
    print("\n" + "=" * 80)
    print(f"TESTING OPTIMIZATION: {optimization_name}")
    print("=" * 80)
    
    for test_case_path in REPRESENTATIVE_CASES:
        test_case = load_test_case(test_case_path)
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        
        case_name = Path(test_case_path).stem
        print(f"\nTesting: {case_name}")
        
        # Test correctness
        try:
            # Replace the function for testing
            original_func = MatrixApp.max_n_subset_intersection_brute_force
            MatrixApp.max_n_subset_intersection_brute_force = optimized_function
            
            res, stats = MatrixApp.run_calculation(
                matrix=matrix,
                params=test_case.params,
                return_stats=True
            )
            
            # Check correctness against expected results
            is_correct = all([
                result == desired for result, desired in zip(res, test_case.expected_results)
            ])
            
            if not is_correct:
                results['correctness'] = False
                results['failures'].append(f"{case_name}: Results mismatch")
                print(f"  ❌ Correctness check FAILED")
                # Restore original function and skip performance testing
                MatrixApp.max_n_subset_intersection_brute_force = original_func
                continue
            else:
                print(f"  ✓ Correctness check passed")
            
            # Test performance
            times = []
            stats_list = []
            
            # Warm-up run
            MatrixApp.run_calculation(
                matrix=matrix,
                params=test_case.params,
                return_stats=True
            )
            
            for _ in range(3):  # Multiple runs for stable timing
                start_time = time.perf_counter()
                res, stats = MatrixApp.run_calculation(
                    matrix=matrix,
                    params=test_case.params,
                    return_stats=True
                )
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                stats_list.append(stats)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            baseline_time = baseline_results[test_case_path]['execution_time']
            speedup = baseline_time / avg_time
            
            results['performance_gains'][test_case_path] = {
                'case_name': case_name,
                'speedup': speedup,
                'baseline_time': baseline_time,
                'optimized_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'all_times': times,
                'stats': stats_list[0] if stats_list else None
            }
            
            results['detailed_results'][test_case_path] = {
                'results': res,
                'stats': stats_list[0] if stats_list else None
            }
            
            print(f"  Baseline time: {baseline_time:.4f}s")
            print(f"  Optimized time: {avg_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Restore original function
            MatrixApp.max_n_subset_intersection_brute_force = original_func
            
        except Exception as e:
            results['correctness'] = False
            results['failures'].append(f"{case_name}: {str(e)}")
            print(f"  ❌ Exception: {str(e)}")
            # Ensure original function is restored
            MatrixApp.max_n_subset_intersection_brute_force = original_func
    
    return results


def validate_optimization(optimization_results: Dict) -> Dict:
    """Validate optimization using all-or-nothing criteria"""
    
    # Criteria 1: All test cases must pass correctness
    correctness_pass = optimization_results['correctness']
    
    # Criteria 2: All test cases must show speedup > 1.0
    all_speedups = [result['speedup'] for result in optimization_results['performance_gains'].values()]
    performance_pass = all(speedup > 1.0 for speedup in all_speedups) if all_speedups else False
    
    # Criteria 3: Average speedup must be > 1.5x
    average_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0.0
    meaningful_improvement = average_speedup > 1.5
    
    return {
        'keep_optimization': correctness_pass and performance_pass and meaningful_improvement,
        'correctness_pass': correctness_pass,
        'performance_pass': performance_pass,
        'average_speedup': average_speedup,
        'min_speedup': min(all_speedups) if all_speedups else 0.0,
        'max_speedup': max(all_speedups) if all_speedups else 0.0,
        'individual_speedups': {
            result['case_name']: result['speedup'] 
            for result in optimization_results['performance_gains'].values()
        }
    }


def generate_validation_report(baseline_results: Dict, optimization_results: Dict, validation_result: Dict, report_name: str = "optimization_validation_report.json") -> Dict:
    """Generate detailed validation report"""
    
    report = {
        'summary': {
            'optimization_name': optimization_results['optimization_name'],
            'optimization_accepted': validation_result['keep_optimization'],
            'correctness_pass': validation_result['correctness_pass'],
            'performance_pass': validation_result['performance_pass'],
            'average_speedup': validation_result['average_speedup'],
            'min_speedup': validation_result['min_speedup'],
            'max_speedup': validation_result['max_speedup']
        },
        'individual_results': validation_result['individual_speedups'],
        'detailed_performance': {
            case_name: {
                'baseline_time': perf['baseline_time'],
                'optimized_time': perf['optimized_time'],
                'speedup': perf['speedup']
            }
            for case_name, perf in [
                (v['case_name'], v) 
                for v in optimization_results['performance_gains'].values()
            ]
        },
        'failures': optimization_results.get('failures', [])
    }
    
    # Save report to file
    with open(report_name, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Optimization: {optimization_results['optimization_name']}")
    print(f"Status: {'✓ ACCEPTED' if validation_result['keep_optimization'] else '✗ REJECTED'}")
    print(f"Correctness: {'✓ PASS' if validation_result['correctness_pass'] else '✗ FAIL'}")
    print(f"Performance: {'✓ PASS' if validation_result['performance_pass'] else '✗ FAIL'}")
    print(f"Average speedup: {validation_result['average_speedup']:.2f}x")
    print(f"Min/Max speedup: {validation_result['min_speedup']:.2f}x / {validation_result['max_speedup']:.2f}x")
    
    if optimization_results.get('failures'):
        print("\nFailures:")
        for failure in optimization_results['failures']:
            print(f"  - {failure}")
    
    return report


def run_optimization_validation_pipeline(optimization_name: str, optimized_function, baseline_results: Optional[Dict] = None) -> Dict:
    """Complete validation pipeline for optimization testing"""
    
    if baseline_results is None:
        print("Phase 1: Establishing baseline...")
        baseline_results = collect_baseline_metrics()
    
    print(f"\nPhase 2: Testing optimization '{optimization_name}'...")
    optimization_results = test_optimization(optimization_name, optimized_function, baseline_results)
    
    print("\nPhase 3: All-or-nothing validation...")
    validation_result = validate_optimization(optimization_results)
    
    # Generate report
    report_name = f"{optimization_name.lower().replace(' ', '_')}_report.json"
    generate_validation_report(baseline_results, optimization_results, validation_result, report_name)
    
    return {
        'baseline': baseline_results,
        'optimization': optimization_results,
        'validation': validation_result
    }


if __name__ == "__main__":
    # Run baseline collection only
    print("Collecting baseline metrics for all SEPARATE mode test cases...")
    baseline = collect_baseline_metrics()
    
    # Save baseline to file for later use
    with open("baseline_metrics.json", "w") as f:
        # Convert non-serializable objects for JSON
        baseline_json = {}
        for key, value in baseline.items():
            baseline_json[key] = {
                k: v for k, v in value.items() 
                if k not in ['results']  # Skip results for JSON serialization
            }
        json.dump(baseline_json, f, indent=2)
    
    print("\nBaseline metrics saved to baseline_metrics.json")