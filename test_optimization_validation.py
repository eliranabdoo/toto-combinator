import os
import time
import json
import pickle
import numpy as np
from pathlib import Path
from core_functions import MatrixApp, CalculationParams, ProcessingMode
from optimized_functions import (
    max_n_subset_intersection_brute_force_with_pruning,
    max_n_subset_intersection_brute_force_with_parallelization
)
from itertools import combinations, product
import heapq
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

# Test data generation
def create_test_matrix(rows=20, cols=8, seed=42):
    """Create a test matrix with controlled data distribution"""
    np.random.seed(seed)
    # Create matrix with some structure to make intersections meaningful
    matrix = np.random.randint(1, 10, size=(rows, cols))
    
    # Add some patterns to create meaningful intersections
    for i in range(rows):
        for j in range(cols):
            if i % 3 == 0:  # Every 3rd row has same values in first 3 columns
                matrix[i, j % 3] = 5
            if i % 5 == 0:  # Every 5th row has same values in last 3 columns  
                matrix[i, 3 + j % 3] = 7
    
    return matrix

def create_test_cases():
    """Create representative test cases for validation"""
    test_cases = []
    
    # Small matrix test cases
    small_matrix = create_test_matrix(rows=15, cols=6, seed=42)
    test_cases.extend([
        {
            'name': 'small_3_separate_0_1',
            'matrix': small_matrix,
            'params': CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=1)
        },
        {
            'name': 'small_4_separate_0_2', 
            'matrix': small_matrix,
            'params': CalculationParams(n=4, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=2)
        },
        {
            'name': 'small_3_separate_1_1',
            'matrix': small_matrix,
            'params': CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=1, top_k=1)
        }
    ])
    
    # Medium matrix test cases
    medium_matrix = create_test_matrix(rows=25, cols=8, seed=123)
    test_cases.extend([
        {
            'name': 'medium_3_separate_0_1',
            'matrix': medium_matrix,
            'params': CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=1)
        },
        {
            'name': 'medium_4_separate_1_2',
            'matrix': medium_matrix,
            'params': CalculationParams(n=4, mode=ProcessingMode.SEPARATE, relaxation=1, top_k=2)
        }
    ])
    
    # Large matrix test cases
    large_matrix = create_test_matrix(rows=40, cols=10, seed=456)
    test_cases.extend([
        {
            'name': 'large_3_separate_0_1',
            'matrix': large_matrix,
            'params': CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=1)
        },
        {
            'name': 'large_4_separate_1_2',
            'matrix': large_matrix,
            'params': CalculationParams(n=4, mode=ProcessingMode.SEPARATE, relaxation=1, top_k=2)
        }
    ])
    
    return test_cases

def collect_baseline_metrics(test_cases):
    """Collect baseline performance metrics for test cases"""
    print("Phase 1: Establishing baseline metrics...")
    baseline_results = {}
    
    for test_case in test_cases:
        print(f"  Running baseline for {test_case['name']}...")
        
        # Run multiple times for more accurate timing
        times = []
        results = None
        
        for _ in range(3):  # 3 runs for average
            start_time = time.perf_counter()
            res = MatrixApp.max_n_subset_intersection_brute_force(
                test_case['matrix'], 
                test_case['params']
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if results is None:
                results = res
        
        avg_time = sum(times) / len(times)
        baseline_results[test_case['name']] = {
            'execution_time': avg_time,
            'results': results,
            'matrix_shape': test_case['matrix'].shape,
            'params': test_case['params']
        }
        
        print(f"    Baseline time: {avg_time:.4f}s")
    
    return baseline_results

def test_optimization(optimization_name, optimized_function, test_cases, baseline_results):
    """Test a single optimization against test cases"""
    print(f"Phase 2: Testing {optimization_name} optimization...")
    results = {
        'correctness': True,
        'performance_gains': {},
        'failures': []
    }
    
    for test_case in test_cases:
        print(f"  Testing {test_case['name']}...")
        
        try:
            # Test correctness
            start_time = time.perf_counter()
            res = optimized_function(test_case['matrix'], test_case['params'])
            end_time = time.perf_counter()
            
            # Compare results with baseline
            baseline_res = baseline_results[test_case['name']]['results']
            is_correct = compare_results(res, baseline_res)
            
            if not is_correct:
                results['correctness'] = False
                results['failures'].append(f"{test_case['name']}: Results don't match baseline")
                print(f"    ❌ Correctness failed")
                continue
            
            # Test performance
            optimized_time = end_time - start_time
            baseline_time = baseline_results[test_case['name']]['execution_time']
            speedup = baseline_time / optimized_time
            
            results['performance_gains'][test_case['name']] = {
                'speedup': speedup,
                'baseline_time': baseline_time,
                'optimized_time': optimized_time
            }
            
            print(f"    ✅ Speedup: {speedup:.2f}x ({baseline_time:.4f}s -> {optimized_time:.4f}s)")
            
        except Exception as e:
            results['correctness'] = False
            results['failures'].append(f"{test_case['name']}: {str(e)}")
            print(f"    ❌ Error: {str(e)}")
    
    return results

def compare_results(res1, res2):
    """Compare two result sets for correctness"""
    if len(res1) != len(res2):
        return False
    
    # Sort both results for comparison
    def sort_key(x):
        return (x[0], x[1], x[2])  # chosen_columns, chosen_rows, chosen_columns_ranges
    
    sorted_res1 = sorted(res1, key=sort_key)
    sorted_res2 = sorted(res2, key=sort_key)
    
    for r1, r2 in zip(sorted_res1, sorted_res2):
        if r1[0] != r2[0] or r1[1] != r2[1] or r1[2] != r2[2]:
            return False
    
    return True

def validate_optimization(optimization_results):
    """Validate optimization using all-or-nothing criteria"""
    print("Phase 3: All-or-nothing validation...")
    
    # Criteria 1: All test cases must pass correctness
    correctness_pass = optimization_results['correctness']
    
    # Criteria 2: All test cases must show speedup > 1.0
    all_speedups = [result['speedup'] for result in optimization_results['performance_gains'].values()]
    performance_pass = all(speedup > 1.0 for speedup in all_speedups) if all_speedups else False
    
    # Criteria 3: Average speedup must be > 1.2x (slightly lower threshold for initial testing)
    average_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    meaningful_improvement = average_speedup > 1.2
    
    validation_result = {
        'keep_optimization': correctness_pass and performance_pass and meaningful_improvement,
        'correctness_pass': correctness_pass,
        'performance_pass': performance_pass,
        'average_speedup': average_speedup,
        'individual_speedups': optimization_results['performance_gains']
    }
    
    print(f"  Correctness: {'✅' if correctness_pass else '❌'}")
    print(f"  Performance: {'✅' if performance_pass else '❌'}")
    print(f"  Average speedup: {average_speedup:.2f}x")
    print(f"  Keep optimization: {'✅' if validation_result['keep_optimization'] else '❌'}")
    
    return validation_result

def generate_validation_report(baseline_results, optimization_results, validation_result, optimization_name):
    """Generate detailed validation report"""
    report = {
        'optimization_name': optimization_name,
        'summary': {
            'optimization_accepted': validation_result['keep_optimization'],
            'correctness_pass': validation_result['correctness_pass'],
            'performance_pass': validation_result['performance_pass'],
            'average_speedup': validation_result['average_speedup']
        },
        'detailed_results': {
            'baseline_performance': baseline_results,
            'optimization_results': optimization_results,
            'validation_result': validation_result
        }
    }
    
    # Save report to file
    report_filename = f'optimization_validation_report_{optimization_name}.json'
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 Report saved to {report_filename}")
    return report

def run_optimization_validation_pipeline(optimization_name, optimized_function):
    """Complete validation pipeline for optimization testing"""
    print(f"\n🚀 Starting validation pipeline for {optimization_name}")
    print("=" * 60)
    
    # Create test cases
    test_cases = create_test_cases()
    print(f"Created {len(test_cases)} test cases")
    
    # Phase 1: Baseline
    baseline_results = collect_baseline_metrics(test_cases)
    
    # Phase 2: Test optimization
    optimization_results = test_optimization(optimization_name, optimized_function, test_cases, baseline_results)
    
    # Phase 3: Validate
    validation_result = validate_optimization(optimization_results)
    
    # Generate report
    report = generate_validation_report(baseline_results, optimization_results, validation_result, optimization_name)
    
    return {
        'baseline': baseline_results,
        'optimization': optimization_results,
        'validation': validation_result,
        'report': report
    }

if __name__ == "__main__":
    print("🚀 Starting comprehensive optimization validation pipeline")
    print("=" * 70)
    
    # Test the baseline function first
    print("\n1. Testing baseline function...")
    test_cases = create_test_cases()
    baseline_results = collect_baseline_metrics(test_cases)
    print("✅ Baseline testing complete!")
    
    # Test pruning optimization
    print("\n2. Testing pruning optimization...")
    pruning_results = run_optimization_validation_pipeline(
        "pruning", 
        max_n_subset_intersection_brute_force_with_pruning
    )
    
    # Test parallelization optimization
    print("\n3. Testing parallelization optimization...")
    parallelization_results = run_optimization_validation_pipeline(
        "parallelization", 
        max_n_subset_intersection_brute_force_with_parallelization
    )
    
    # Generate final summary report
    print("\n4. Generating final summary report...")
    final_report = {
        'baseline_results': baseline_results,
        'pruning_optimization': pruning_results,
        'parallelization_optimization': parallelization_results,
        'summary': {
            'pruning_accepted': pruning_results['validation']['keep_optimization'],
            'parallelization_accepted': parallelization_results['validation']['keep_optimization'],
            'pruning_speedup': pruning_results['validation']['average_speedup'],
            'parallelization_speedup': parallelization_results['validation']['average_speedup']
        }
    }
    
    with open('final_optimization_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print("\n📊 Final Summary:")
    print(f"  Pruning optimization: {'✅ ACCEPTED' if final_report['summary']['pruning_accepted'] else '❌ REJECTED'}")
    print(f"    Average speedup: {final_report['summary']['pruning_speedup']:.2f}x")
    print(f"  Parallelization optimization: {'✅ ACCEPTED' if final_report['summary']['parallelization_accepted'] else '❌ REJECTED'}")
    print(f"    Average speedup: {final_report['summary']['parallelization_speedup']:.2f}x")
    print("\n📁 Detailed reports saved:")
    print("  - optimization_validation_report_pruning.json")
    print("  - optimization_validation_report_parallelization.json") 
    print("  - final_optimization_report.json")