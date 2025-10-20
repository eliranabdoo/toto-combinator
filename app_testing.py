"""
Stripped down version of app.py for testing purposes (without tkinter dependencies)
"""

import logging
import math
import time
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from itertools import combinations, groupby, product
from datetime import datetime
import heapq
from data_models import CalculationParams, CalculationResult, ProcessingMode, TotoResult


def calculation_to_toto_result(
    calculation_result: CalculationResult, mode: ProcessingMode
) -> TotoResult:
    sikum = None
    if all([p[1] == p[0] for p in calculation_result.chosen_columns_ranges]):
        sikum = sum([p[0] for p in calculation_result.chosen_columns_ranges])
    return TotoResult(
        turim=[c + 1 for c in calculation_result.chosen_columns],
        mahzorim=[r + 1 for r in calculation_result.chosen_rows],
        tvachim=[
            f"{start}-{end}" if start != end else str(start)
            for start, end in calculation_result.chosen_columns_ranges
        ]
        * (
            len(calculation_result.chosen_columns)
            if mode == ProcessingMode.AGGREGATIVE
            else 1
        ),
        sikum=sikum,
    )


class MatrixApp:
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
    def filter_columns_by_ranges(matrix, ranges):
        filtered_columns = []
        for col_idx, (min_val, max_val) in enumerate(ranges):
            col = matrix[:, col_idx]
            indices = {
                row_idx for row_idx, x in enumerate(col) if min_val <= x <= max_val
            }
            filtered_columns.append(indices)
        return filtered_columns

    @staticmethod
    def max_n_representative_intersection(
        groups, n, top_k, stats: Optional[dict], enable_pruning: bool = False
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
        stats["pruned_choices"] = 0
        stats["pruned_combinations"] = 0
        max_choices_time = 0.0
        
        # Track minimum threshold for pruning (only used when pruning is enabled)
        min_threshold = 0 if enable_pruning else -1

        # All combinations of n groups out of N
        t0 = time.perf_counter()
        for group_indices in combinations(range(len(groups)), n):  # O(C^n)
            selected_groups = [groups[i] for i in group_indices]
            # All ways to pick one subset from each selected group

            num_choices = math.prod(len(g) for g in selected_groups)
            stats["total_choices"] += num_choices
            stats["max_choices_per_combination"] = max(
                stats["max_choices_per_combination"], num_choices
            )
            t2 = time.perf_counter()
            choices_pruned = 0
            
            for choice in product(
                *[range(len(g)) for g in selected_groups]
            ):  # O(max(groups(R))^n)
                selected_subsets = [selected_groups[i][choice[i]] for i in range(n)]
                current_intersection = selected_subsets[0]
                
                # Early pruning based on initial set size (only when pruning enabled)
                if enable_pruning and len(current_intersection) < min_threshold:
                    choices_pruned += 1
                    stats["pruned_choices"] += 1
                    continue
                
                # Compute intersection (with optional early termination)
                pruned = False
                for subset in selected_subsets[1:]:
                    current_intersection = current_intersection.intersection(subset)
                    
                    # Early termination if intersection becomes too small (only when pruning enabled)
                    if enable_pruning and len(current_intersection) < min_threshold:
                        choices_pruned += 1
                        stats["pruned_choices"] += 1
                        pruned = True
                        break
                
                if pruned:
                    continue

                score = len(current_intersection)
                
                # Only consider adding if score is good enough (or pruning disabled)
                if not enable_pruning or score >= min_threshold:
                    # Use a min-heap of size top_k
                    item = (score, group_indices, sorted(current_intersection))
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                        # Update threshold when heap becomes full (only when pruning enabled)
                        if enable_pruning and len(heap) == top_k:
                            min_threshold = heap[0][0]
                    else:
                        if item > heap[0]:
                            heapq.heappushpop(heap, item)
                            # Update threshold with new minimum (only when pruning enabled)
                            if enable_pruning:
                                min_threshold = heap[0][0]
            
            if choices_pruned == num_choices:
                stats["pruned_combinations"] += 1
                
            t3 = time.perf_counter()
            max_choices_time = max(max_choices_time, t3 - t2)
        t1 = time.perf_counter()
        stats["combinations_time_s"] = t1 - t0
        stats["max_choices_time"] = max_choices_time
        stats["average_choices_per_combination"] = (
            stats["total_choices"] / stats["combinations"]
        )
        if enable_pruning and stats["total_choices"] > 0:
            stats["pruning_rate"] = stats["pruned_choices"] / stats["total_choices"]

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

        top_intersections = MatrixApp.max_n_representative_intersection(
            per_col_subsets, n=params.n, top_k=params.top_k, stats=stats, 
            enable_pruning=params.enable_pruning
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
    def max_n_subset_sums_brute_force(
        matrix: np.ndarray, params: CalculationParams, stats: Optional[Dict]
    ) -> List[CalculationResult]:
        if params.relaxation > 0:
            raise NotImplementedError("כרגע אין תמיכה בקפיצות במצב חישוב סיכומי")
        n = params.n
        top_k = params.top_k
        heap = []


        stats["num_combinations"] = matrix.shape[1] ** n

        t0 = time.perf_counter()
        for combo_indices in combinations(range(matrix.shape[1]), n):
            sums: np.ndarray = matrix[:, combo_indices].sum(axis=1)
            counts = np.bincount(sums)
            max_s = np.argmax(counts)
            max_count = counts[max_s]
            chosen_rows = sorted(list(np.nonzero((sums == max_s)))[0])
            chosen_rows_range = [(max_s, max_s)]
            item = (max_count, combo_indices, chosen_rows, chosen_rows_range)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if item > heap[0]:
                    heapq.heappushpop(heap, item)

        # Sort results by count descending, then by columns
        results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)

        t1 = time.perf_counter()
        stats["total_time_elapsed_s"] = t1 - t0
        return [
            CalculationResult(
                chosen_columns=list(combo_indices),
                chosen_rows=chosen_rows,
                chosen_columns_ranges=chosen_rows_range,
            )
            for _, combo_indices, chosen_rows, chosen_rows_range in results
        ]

    @staticmethod
    def run_calculation(
        matrix, params: CalculationParams, return_stats: bool = False
    ) -> Tuple[List[CalculationResult], Optional[Dict]]:
        if params.mode == ProcessingMode.SEPARATE:
            calculation_func = MatrixApp.max_n_subset_intersection_brute_force
        elif params.mode == ProcessingMode.AGGREGATIVE:
            calculation_func = MatrixApp.max_n_subset_sums_brute_force

        stats = {} if return_stats else None
        res = calculation_func(matrix, params, stats)
        return res, stats