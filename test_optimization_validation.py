import os
import time
import json
import pickle
import copy
from pathlib import Path
from app import MatrixApp
from data_models import MatrixAppTestCase, ProcessingMode, CalculationParams
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Select only SEPARATE mode test cases as representatives
REPRESENTATIVE_CASES = [
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_0_1.pickle",  # Small matrix, n=3, no relaxation
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_0_2.pickle",  # Small matrix, n=4, no relaxation
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_1_2.pickle",  # Small matrix, n=3, with relaxation
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_1_1.pickle",  # Small matrix, n=4, with relaxation
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_0_1.pickle",  # Large matrix, n=3, no relaxation
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_0_2.pickle",  # Large matrix, n=4, no relaxation
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_1_2.pickle",  # Large matrix, n=3, with relaxation
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_1_1.pickle",  # Large matrix, n=4, with relaxation
]


def load_test_case(test_case_path: str) -> MatrixAppTestCase:
    """Load a test case from pickle file"""
    return pickle.load(open(test_case_path, "rb"), fix_imports=False)


def collect_baseline_metrics(repeat_count: int = 3) -> Dict:
    """Collect baseline performance metrics for representative test cases"""
    print("Collecting baseline metrics...")
    baseline_results = {}
    
    for test_case_path in REPRESENTATIVE_CASES:
        print(f"  Processing {os.path.basename(test_case_path)}...")
        test_case = load_test_case(test_case_path)
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        
        # Run multiple times for more stable timing
        times = []
        stats_collection = []
        
        for _ in range(repeat_count):
            start_time = time.perf_counter()
            res, stats = MatrixApp.run_calculation(
                matrix=matrix,
                params=test_case.params,
                return_stats=True
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            stats_collection.append(stats)
        
        baseline_results[test_case_path] = {
            'execution_times': times,
            'avg_execution_time': sum(times) / len(times),
            'min_execution_time': min(times),
            'max_execution_time': max(times),
            'stats': stats_collection[0],  # Stats should be the same each run
            'matrix_shape': matrix.shape,
            'params': {
                'n': test_case.params.n,
                'mode': test_case.params.mode.value,
                'relaxation': test_case.params.relaxation,
                'top_k': test_case.params.top_k
            }
        }
    
    return baseline_results


def test_optimization_correctness(optimized_function, test_case_path: str) -> Tuple[bool, Optional[str]]:
    """Test correctness of an optimization against a single test case"""
    try:
        test_case = load_test_case(test_case_path)
        matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
        
        # Run optimized function
        res = optimized_function(matrix, test_case.params, stats=None)
        
        # Check correctness
        is_correct = all([
            result == desired for result, desired in zip(res, test_case.expected_results)
        ])
        
        if not is_correct:
            # Provide detailed error info
            for i, (result, desired) in enumerate(zip(res, test_case.expected_results)):
                if result != desired:
                    return False, f"Result {i} mismatch: got {result}, expected {desired}"
        
        return True, None
        
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_optimization_performance(optimized_function, test_case_path: str, 
                                 baseline_time: float, repeat_count: int = 3) -> Dict:
    """Test performance of an optimization against a single test case"""
    test_case = load_test_case(test_case_path)
    matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
    
    times = []
    stats_collection = []
    
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        res = optimized_function(matrix, test_case.params, stats={})
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    speedup = baseline_time / avg_time
    
    return {
        'execution_times': times,
        'avg_execution_time': avg_time,
        'min_execution_time': min(times),
        'max_execution_time': max(times),
        'baseline_time': baseline_time,
        'speedup': speedup,
        'improvement_percentage': (speedup - 1) * 100
    }


def test_optimization(optimization_name: str, optimized_function, 
                     baseline_results: Dict, repeat_count: int = 3) -> Dict:
    """Test a single optimization against all representative test cases"""
    print(f"\nTesting optimization: {optimization_name}")
    results = {
        'name': optimization_name,
        'correctness': True,
        'test_results': {},
        'failures': [],
        'summary': {}
    }
    
    all_speedups = []
    
    for test_case_path in REPRESENTATIVE_CASES:
        test_name = os.path.basename(test_case_path)
        print(f"  Testing {test_name}...")
        
        # Test correctness
        is_correct, error_msg = test_optimization_correctness(optimized_function, test_case_path)
        
        if not is_correct:
            results['correctness'] = False
            results['failures'].append(f"{test_name}: {error_msg}")
            print(f"    ❌ Correctness check failed: {error_msg}")
            continue
        
        print(f"    ✓ Correctness check passed")
        
        # Test performance
        baseline_time = baseline_results[test_case_path]['avg_execution_time']
        perf_results = test_optimization_performance(
            optimized_function, test_case_path, baseline_time, repeat_count
        )
        
        results['test_results'][test_case_path] = perf_results
        all_speedups.append(perf_results['speedup'])
        
        print(f"    ✓ Performance: {perf_results['speedup']:.2f}x speedup")
    
    # Calculate summary statistics
    if all_speedups:
        results['summary'] = {
            'average_speedup': sum(all_speedups) / len(all_speedups),
            'min_speedup': min(all_speedups),
            'max_speedup': max(all_speedups),
            'all_tests_passed': results['correctness'],
            'tests_passed': len(all_speedups),
            'tests_failed': len(results['failures'])
        }
    
    return results


def validate_optimization(optimization_results: Dict) -> Dict:
    """Validate optimization using all-or-nothing criteria"""
    
    # Criteria 1: All test cases must pass correctness
    correctness_pass = optimization_results['correctness']
    
    # Criteria 2: All test cases must show speedup > 1.0
    all_speedups = [result['speedup'] for result in optimization_results['test_results'].values()]
    performance_pass = all(speedup > 1.0 for speedup in all_speedups) if all_speedups else False
    
    # Criteria 3: Average speedup must be > 1.5x for meaningful improvement
    average_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    meaningful_improvement = average_speedup > 1.5
    
    return {
        'keep_optimization': correctness_pass and performance_pass and meaningful_improvement,
        'correctness_pass': correctness_pass,
        'performance_pass': performance_pass,
        'meaningful_improvement': meaningful_improvement,
        'average_speedup': average_speedup,
        'decision': 'ACCEPTED' if (correctness_pass and performance_pass and meaningful_improvement) else 'REJECTED'
    }


def generate_validation_report(baseline_results: Dict, optimization_results: List[Dict], 
                              validation_results: List[Dict]) -> Dict:
    """Generate comprehensive validation report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'test_cases': len(baseline_results),
            'details': {}
        },
        'optimizations': [],
        'summary': {
            'total_optimizations_tested': len(optimization_results),
            'accepted_optimizations': sum(1 for v in validation_results if v['keep_optimization']),
            'rejected_optimizations': sum(1 for v in validation_results if not v['keep_optimization'])
        }
    }
    
    # Add baseline details
    for test_path, baseline_data in baseline_results.items():
        test_name = os.path.basename(test_path)
        report['baseline']['details'][test_name] = {
            'avg_execution_time': baseline_data['avg_execution_time'],
            'matrix_shape': baseline_data['matrix_shape'],
            'params': baseline_data['params'],
            'stats': baseline_data['stats']
        }
    
    # Add optimization results
    for opt_result, val_result in zip(optimization_results, validation_results):
        opt_report = {
            'name': opt_result['name'],
            'validation': val_result,
            'test_results': {}
        }
        
        for test_path, test_result in opt_result['test_results'].items():
            test_name = os.path.basename(test_path)
            opt_report['test_results'][test_name] = {
                'speedup': test_result['speedup'],
                'improvement_percentage': test_result['improvement_percentage'],
                'avg_execution_time': test_result['avg_execution_time'],
                'baseline_time': test_result['baseline_time']
            }
        
        report['optimizations'].append(opt_report)
    
    return report


def save_report(report: Dict, filename: str = 'optimization_validation_report.json'):
    """Save report to file and print summary"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total optimizations tested: {report['summary']['total_optimizations_tested']}")
    print(f"Accepted optimizations: {report['summary']['accepted_optimizations']}")
    print(f"Rejected optimizations: {report['summary']['rejected_optimizations']}")
    print()
    
    for opt in report['optimizations']:
        status = "✅ ACCEPTED" if opt['validation']['decision'] == 'ACCEPTED' else "❌ REJECTED"
        print(f"{status} - {opt['name']}")
        print(f"  Average speedup: {opt['validation']['average_speedup']:.2f}x")
        if opt['test_results']:
            speedups = [r['speedup'] for r in opt['test_results'].values()]
            print(f"  Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print()
    
    print(f"Full report saved to: {filename}")
    print(f"{'='*60}")


def run_optimization_validation_pipeline(optimizations_to_test: List[Tuple[str, callable]], 
                                        repeat_count: int = 3) -> Dict:
    """Complete validation pipeline for optimization testing"""
    
    print("="*60)
    print("STARTING OPTIMIZATION VALIDATION PIPELINE")
    print("="*60)
    
    # Phase 1: Establish baseline
    print("\nPhase 1: Establishing baseline...")
    baseline_results = collect_baseline_metrics(repeat_count)
    print(f"  ✓ Baseline collected for {len(baseline_results)} test cases")
    
    # Phase 2: Test optimizations
    print("\nPhase 2: Testing optimizations...")
    optimization_results = []
    validation_results = []
    
    for opt_name, opt_func in optimizations_to_test:
        opt_result = test_optimization(opt_name, opt_func, baseline_results, repeat_count)
        optimization_results.append(opt_result)
        
        val_result = validate_optimization(opt_result)
        validation_results.append(val_result)
    
    # Phase 3: Generate report
    print("\nPhase 3: Generating report...")
    report = generate_validation_report(baseline_results, optimization_results, validation_results)
    save_report(report)
    
    return {
        'baseline': baseline_results,
        'optimizations': optimization_results,
        'validations': validation_results,
        'report': report
    }


if __name__ == "__main__":
    # This will be used to test optimizations
    # Example usage:
    # from app import MatrixApp
    # optimizations_to_test = [
    #     ("Baseline (no changes)", MatrixApp.max_n_subset_intersection_brute_force),
    # ]
    # results = run_optimization_validation_pipeline(optimizations_to_test)
    
    print("Test framework ready. Import this module to test your optimizations.")