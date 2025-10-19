#!/usr/bin/env python3

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from itertools import combinations, product, groupby
import heapq
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import data models
from data_models import MatrixAppTestCase, ProcessingMode, CalculationParams, CalculationResult


class MatrixAppOptimization:
    """Stripped down version of MatrixApp for testing optimizations"""
    
    @staticmethod
    def load_matrix_from_excel(file_path) -> np.ndarray:
        df = pd.read_excel(file_path, header=None)
        c = df.isna().iloc[:, 0]
        r = df.isna().iloc[0]
        c_min = r[r].index[0] if sum(r) > 0 else None
        r_min = c[c].index[0] if sum(c) > 0 else None
        df = df.iloc[:r_min, :c_min].dropna().astype(int)
        return df.values

    @staticmethod
    def max_n_representative_intersection(
        groups, n, top_k, stats: Optional[dict]
    ) -> List[Tuple[list[int], list[int]]]:
        """
        groups: O(C*R)
        """

        heap = []
        if stats is None:
            stats = {}

        stats["combinations"] = len(groups) ** n
        stats["total_choices"] = 0
        stats["max_choices_per_combination"] = 0
        max_choices_time = 0.0

        # All combinations of n groups out of N
        t0 = time.perf_counter()
        for group_indices in combinations(range(len(groups)), n):  # O(C^n)
            selected_groups = [groups[i] for i in group_indices]
            # All ways to pick one subset from each selected group

            num_choices = 1
            for g in selected_groups:
                num_choices *= len(g)
            stats["total_choices"] += num_choices
            stats["max_choices_per_combination"] = max(
                stats["max_choices_per_combination"], num_choices
            )
            t2 = time.perf_counter()
            for choice in product(
                *[range(len(g)) for g in selected_groups]
            ):  # O(max(groups(R))^n)
                selected_subsets = [selected_groups[i][choice[i]] for i in range(n)]
                current_intersection = selected_subsets[0]
                for subset in selected_subsets[1:]:
                    current_intersection = current_intersection.intersection(subset)

                score = len(current_intersection)
                # Use a min-heap of size top_k
                item = (score, group_indices, sorted(current_intersection))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    if item > heap[0]:
                        heapq.heappushpop(heap, item)
            t3 = time.perf_counter()
            max_choices_time = max(max_choices_time, t3 - t2)
        t1 = time.perf_counter()
        stats["combinations_time_s"] = t1 - t0
        stats["max_choices_time"] = max_choices_time
        stats["average_choices_per_combination"] = (
            stats["total_choices"] / stats["combinations"]
        )

        # Sort results by intersection size descending, then by group indices
        results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
        return [
            (list(group_indices), intersection)
            for _, group_indices, intersection in results
        ]

    @staticmethod
    def max_n_subset_intersection_brute_force(
        matrix: np.ndarray, params: CalculationParams, stats: Optional[Dict]
    ) -> List[CalculationResult]:
        per_col_subsets = []

        for col in matrix.T:  # O(C)
            index_and_elem_pairs = sorted(
                list(enumerate(col)), key=lambda p: p[1]
            )  # O(RlogR)
            groups = [
                list(g) for _, g in groupby(index_and_elem_pairs, key=lambda p: p[1])
            ]  # O(R)
            if params.relaxation > 0:
                relaxed_groups = []
                for group_index, group in enumerate(groups):
                    relaxed_group = group.copy()
                    relaxed_groups.append(relaxed_group)
                    if len(group) == 0:
                        continue
                    group_value = group[0][1]
                    for added_group in groups[group_index + 1 :]:
                        if (added_group[0][1] - group_value) > params.relaxation:
                            break
                        relaxed_group.extend(added_group.copy())
                groups = relaxed_groups
            subsets = [set([p[0] for p in group]) for group in groups]
            per_col_subsets.append(subsets)

        top_intersections = MatrixAppOptimization.max_n_representative_intersection(
            per_col_subsets, n=params.n, top_k=params.top_k, stats=stats
        )

        res = []
        for intersection in top_intersections:
            chosen_columns, chosen_rows = intersection
            chosen_columns_ranges = []

            for col in chosen_columns:
                view = matrix[chosen_rows, col]
                col_range = (view.min(), view.max())
                chosen_columns_ranges.append(col_range)
                assert col_range[1] - col_range[0] <= params.relaxation, (
                    "invalid relaxation"
                )
            res.append(
                CalculationResult(
                    chosen_columns=chosen_columns,
                    chosen_rows=chosen_rows,
                    chosen_columns_ranges=chosen_columns_ranges,
                )
            )

        return res

    @staticmethod
    def run_calculation(
        matrix, params: CalculationParams, return_stats: bool = False
    ) -> Tuple[List[CalculationResult], Optional[Dict]]:
        if params.mode == ProcessingMode.SEPARATE:
            stats = {} if return_stats else None
            res = MatrixAppOptimization.max_n_subset_intersection_brute_force(matrix, params, stats)
            return res, stats
        else:
            raise NotImplementedError("Only SEPARATE mode is being optimized")


# Test cases for SEPARATE mode only
REPRESENTATIVE_CASES = [
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_0_1.pickle",  
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_0_2.pickle",  
    "tests/assets/ground_truths/131_3_ProcessingMode.SEPARATE_1_2.pickle",  
    "tests/assets/ground_truths/131_4_ProcessingMode.SEPARATE_1_1.pickle",  
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_0_1.pickle",  
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_0_2.pickle",  
    "tests/assets/ground_truths/243_3_ProcessingMode.SEPARATE_1_2.pickle",  
    "tests/assets/ground_truths/243_4_ProcessingMode.SEPARATE_1_1.pickle",  
]


def load_test_case(test_case_path: str) -> MatrixAppTestCase:
    """Load a test case from pickle file"""
    return pickle.load(open(test_case_path, "rb"), fix_imports=False)


def collect_baseline_metrics(repeat_count: int = 3) -> Dict:
    """Collect baseline performance metrics"""
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE METRICS")
    print("="*60)
    
    baseline_results = {}
    
    for test_case_path in REPRESENTATIVE_CASES:
        test_name = os.path.basename(test_case_path)
        print(f"\nProcessing {test_name}...")
        test_case = load_test_case(test_case_path)
        matrix = MatrixAppOptimization.load_matrix_from_excel(test_case.matrix_file_path)
        
        # Run multiple times for stable timing
        times = []
        stats_collection = []
        
        for run in range(repeat_count):
            start_time = time.perf_counter()
            res, stats = MatrixAppOptimization.run_calculation(
                matrix=matrix,
                params=test_case.params,
                return_stats=True
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            if run == 0:
                stats_collection.append(stats)
            print(f"  Run {run+1}: {elapsed:.4f}s")
        
        avg_time = sum(times) / len(times)
        
        baseline_results[test_case_path] = {
            'execution_times': times,
            'avg_execution_time': avg_time,
            'min_execution_time': min(times),
            'max_execution_time': max(times),
            'stats': stats_collection[0],
            'matrix_shape': matrix.shape,
            'params': {
                'n': test_case.params.n,
                'mode': test_case.params.mode.value,
                'relaxation': test_case.params.relaxation,
                'top_k': test_case.params.top_k
            }
        }
        
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Params: n={test_case.params.n}, relaxation={test_case.params.relaxation}, top_k={test_case.params.top_k}")
        if stats_collection[0]:
            print(f"  Total combinations: {stats_collection[0].get('combinations', 'N/A')}")
            print(f"  Total choices: {stats_collection[0].get('total_choices', 'N/A')}")
    
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    
    # Group by matrix file
    by_matrix = {}
    for path, data in baseline_results.items():
        matrix_file = '131.xlsx' if '131' in path else '243.xlsx'
        if matrix_file not in by_matrix:
            by_matrix[matrix_file] = []
        by_matrix[matrix_file].append((path, data))
    
    for matrix_file, cases in by_matrix.items():
        print(f"\n{matrix_file} (shape: {cases[0][1]['matrix_shape']}):")
        total_time = 0
        for path, data in cases:
            test_name = os.path.basename(path)
            params_str = f"n={data['params']['n']}, relax={data['params']['relaxation']}, top_k={data['params']['top_k']}"
            print(f"  {test_name:60s} {data['avg_execution_time']:8.4f}s  ({params_str})")
            total_time += data['avg_execution_time']
        print(f"  {'Total time for ' + matrix_file:60s} {total_time:8.4f}s")
    
    total_baseline_time = sum(data['avg_execution_time'] for data in baseline_results.values())
    print(f"\n{'TOTAL BASELINE TIME':60s} {total_baseline_time:8.4f}s")
    
    return baseline_results


if __name__ == "__main__":
    baseline = collect_baseline_metrics(repeat_count=3)
    
    # Save baseline to file
    with open('baseline_metrics.json', 'w') as f:
        # Convert stats to be JSON serializable
        serializable = {}
        for k, v in baseline.items():
            serializable[k] = {**v}
            serializable[k]['stats'] = {str(sk): sv for sk, sv in v['stats'].items()} if v['stats'] else None
        json.dump(serializable, f, indent=2)
    
    print(f"\nBaseline metrics saved to baseline_metrics.json")