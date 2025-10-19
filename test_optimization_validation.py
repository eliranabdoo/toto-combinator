#!/usr/bin/env python3
"""
Optimization Validation Framework for max_n_subset_intersection_brute_force

This module provides comprehensive testing and validation for algorithmic optimizations
of the SEPARATE processing mode in the MatrixApp application.
"""

import os
import time
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from app import MatrixApp, CalculationParams, ProcessingMode

@dataclass
class TestCase:
    """Represents a single test case for optimization validation"""
    name: str
    matrix: np.ndarray
    params: CalculationParams
    expected_results: List[Tuple[list[int], list[int], list[Tuple[int, int]]]]

@dataclass
class OptimizationResult:
    """Results from testing an optimization"""
    optimization_name: str
    correctness_pass: bool
    performance_gains: Dict[str, Dict[str, float]]
    failures: List[str]
    average_speedup: float

class OptimizationValidator:
    """Main class for validating algorithmic optimizations"""
    
    def __init__(self):
        self.baseline_results = {}
        self.test_cases = self._create_representative_test_cases()
    
    def _create_representative_test_cases(self) -> List[TestCase]:
        """Create representative test cases covering different scenarios"""
        
        # Test Case 1: Small matrix, n=3, no relaxation
        matrix1 = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [2, 3, 4, 6],
            [1, 3, 4, 7],
            [2, 4, 5, 8]
        ])
        
        # Test Case 2: Small matrix, n=4, with relaxation
        matrix2 = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [2, 3, 4, 5, 7],
            [1, 3, 4, 5, 8],
            [2, 4, 5, 6, 9],
            [3, 5, 6, 7, 10]
        ])
        
        # Test Case 3: Medium matrix, n=3, no relaxation
        matrix3 = np.random.randint(1, 10, size=(15, 8))
        
        # Test Case 4: Medium matrix, n=4, with relaxation
        matrix4 = np.random.randint(1, 8, size=(12, 10))
        
        test_cases = [
            TestCase(
                name="small_matrix_n3_no_relaxation",
                matrix=matrix1,
                params=CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=2),
                expected_results=[]
            ),
            TestCase(
                name="small_matrix_n4_with_relaxation",
                matrix=matrix2,
                params=CalculationParams(n=4, mode=ProcessingMode.SEPARATE, relaxation=1, top_k=2),
                expected_results=[]
            ),
            TestCase(
                name="medium_matrix_n3_no_relaxation",
                matrix=matrix3,
                params=CalculationParams(n=3, mode=ProcessingMode.SEPARATE, relaxation=0, top_k=3),
                expected_results=[]
            ),
            TestCase(
                name="medium_matrix_n4_with_relaxation",
                matrix=matrix4,
                params=CalculationParams(n=4, mode=ProcessingMode.SEPARATE, relaxation=1, top_k=3),
                expected_results=[]
            )
        ]
        
        # Generate expected results for each test case
        for test_case in test_cases:
            test_case.expected_results = MatrixApp.max_n_subset_intersection_brute_force(
                test_case.matrix, test_case.params
            )
        
        return test_cases
    
    def collect_baseline_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect baseline performance metrics for all test cases"""
        print("Phase 1: Collecting baseline metrics...")
        baseline_results = {}
        
        for test_case in self.test_cases:
            print(f"  Running baseline for {test_case.name}...")
            
            # Run multiple times for more accurate timing
            times = []
            for _ in range(3):
                start_time = time.perf_counter()
                MatrixApp.max_n_subset_intersection_brute_force(test_case.matrix, test_case.params)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            baseline_results[test_case.name] = {
                'execution_time': np.mean(times),
                'execution_time_std': np.std(times),
                'matrix_shape': test_case.matrix.shape,
                'params': test_case.params
            }
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def test_optimization(self, optimization_name: str, optimized_function) -> OptimizationResult:
        """Test a single optimization against all test cases"""
        print(f"Phase 2: Testing {optimization_name} optimization...")
        
        results = {
            'correctness': True,
            'performance_gains': {},
            'failures': []
        }
        
        for test_case in self.test_cases:
            print(f"  Testing {test_case.name}...")
            
            try:
                # Test correctness
                start_time = time.perf_counter()
                optimized_results = optimized_function(test_case.matrix, test_case.params)
                end_time = time.perf_counter()
                
                # Check correctness by comparing results
                is_correct = self._compare_results(optimized_results, test_case.expected_results)
                
                if not is_correct:
                    results['correctness'] = False
                    results['failures'].append(f"{test_case.name}: Results don't match expected")
                    print(f"    ❌ Correctness failed for {test_case.name}")
                else:
                    print(f"    ✅ Correctness passed for {test_case.name}")
                
                # Test performance
                optimized_time = end_time - start_time
                baseline_time = self.baseline_results[test_case.name]['execution_time']
                speedup = baseline_time / optimized_time if optimized_time > 0 else 0
                
                results['performance_gains'][test_case.name] = {
                    'speedup': speedup,
                    'baseline_time': baseline_time,
                    'optimized_time': optimized_time,
                    'improvement_percent': ((baseline_time - optimized_time) / baseline_time) * 100
                }
                
                print(f"    Speedup: {speedup:.2f}x ({results['performance_gains'][test_case.name]['improvement_percent']:.1f}% improvement)")
                
            except Exception as e:
                results['correctness'] = False
                results['failures'].append(f"{test_case.name}: {str(e)}")
                print(f"    ❌ Error in {test_case.name}: {str(e)}")
        
        # Calculate average speedup
        if results['performance_gains']:
            speedups = [result['speedup'] for result in results['performance_gains'].values()]
            average_speedup = np.mean(speedups)
        else:
            average_speedup = 0.0
        
        return OptimizationResult(
            optimization_name=optimization_name,
            correctness_pass=results['correctness'],
            performance_gains=results['performance_gains'],
            failures=results['failures'],
            average_speedup=average_speedup
        )
    
    def _compare_results(self, result1: List, result2: List) -> bool:
        """Compare two result lists for correctness"""
        if len(result1) != len(result2):
            return False
        
        # Sort both results for comparison
        sorted1 = sorted(result1, key=lambda x: (x[0], x[1]))
        sorted2 = sorted(result2, key=lambda x: (x[0], x[1]))
        
        for r1, r2 in zip(sorted1, sorted2):
            if r1[0] != r2[0] or r1[1] != r2[1] or r1[2] != r2[2]:
                return False
        
        return True
    
    def validate_optimization(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """Validate optimization using all-or-nothing criteria"""
        
        # Criteria 1: All test cases must pass correctness
        correctness_pass = optimization_result.correctness_pass
        
        # Criteria 2: All test cases must show speedup > 1.0
        all_speedups = [result['speedup'] for result in optimization_result.performance_gains.values()]
        performance_pass = all(speedup > 1.0 for speedup in all_speedups) if all_speedups else False
        
        # Criteria 3: Average speedup must be > 1.5x
        meaningful_improvement = optimization_result.average_speedup > 1.5
        
        return {
            'keep_optimization': correctness_pass and performance_pass and meaningful_improvement,
            'correctness_pass': correctness_pass,
            'performance_pass': performance_pass,
            'meaningful_improvement': meaningful_improvement,
            'average_speedup': optimization_result.average_speedup,
            'individual_speedups': optimization_result.performance_gains
        }
    
    def generate_report(self, optimization_results: List[OptimizationResult]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        report = {
            'summary': {
                'total_optimizations_tested': len(optimization_results),
                'successful_optimizations': sum(1 for r in optimization_results if r.correctness_pass),
                'baseline_metrics': self.baseline_results
            },
            'optimization_details': {}
        }
        
        for result in optimization_results:
            validation = self.validate_optimization(result)
            
            report['optimization_details'][result.optimization_name] = {
                'validation': validation,
                'performance_gains': result.performance_gains,
                'failures': result.failures,
                'average_speedup': result.average_speedup
            }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = "optimization_validation_report.json"):
        """Save optimization report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {filename}")

def main():
    """Main function to run optimization validation"""
    validator = OptimizationValidator()
    
    # Collect baseline metrics
    baseline = validator.collect_baseline_metrics()
    print(f"\nBaseline metrics collected for {len(baseline)} test cases")
    
    # This will be populated with actual optimization functions
    optimization_results = []
    
    # Generate initial report
    report = validator.generate_report(optimization_results)
    validator.save_report(report)
    
    print("\nOptimization validation framework ready!")
    print("Baseline metrics:")
    for name, metrics in baseline.items():
        print(f"  {name}: {metrics['execution_time']:.4f}s ± {metrics['execution_time_std']:.4f}s")

if __name__ == "__main__":
    main()