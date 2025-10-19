"""
Pruning optimization for max_n_subset_intersection_brute_force

This module implements the pruning optimization which uses early termination
to skip combinations that cannot improve the current top-k results.
"""

import math
import time
import heapq
from typing import Dict, List, Optional, Tuple
from itertools import combinations, groupby, product
import numpy as np

from data_models import CalculationParams, CalculationResult


class MatrixAppPruning:
    @staticmethod
    def max_n_representative_intersection_with_pruning(
        groups, n, top_k, stats: Optional[dict]
    ) -> List[Tuple[list[int], list[int]]]:
        """
        Optimized version with pruning based on early termination.
        
        Key optimization:
        - Incrementally compute intersection and stop early if it becomes too small
        - Dynamically update pruning threshold based on current heap minimum
        """
        heap = []
        if stats is None:
            stats = {}

        stats["combinations"] = len(groups) ** n
        stats["total_choices"] = 0
        stats["max_choices_per_combination"] = 0
        stats["pruned_combinations"] = 0
        stats["pruned_choices"] = 0
        max_choices_time = 0.0

        # Track minimum threshold for pruning
        min_threshold = 0

        # All combinations of n groups out of N
        t0 = time.perf_counter()
        for group_indices in combinations(range(len(groups)), n):  # O(C^n)
            selected_groups = [groups[i] for i in group_indices]
            
            num_choices = math.prod(len(g) for g in selected_groups)
            stats["total_choices"] += num_choices
            stats["max_choices_per_combination"] = max(
                stats["max_choices_per_combination"], num_choices
            )
            
            t2 = time.perf_counter()
            choices_processed = 0
            choices_pruned = 0
            
            for choice in product(*[range(len(g)) for g in selected_groups]):
                selected_subsets = [selected_groups[i][choice[i]] for i in range(n)]
                
                # Incremental intersection with early termination
                current_intersection = selected_subsets[0]
                
                # Early pruning based on initial set size
                if len(current_intersection) < min_threshold:
                    choices_pruned += 1
                    continue
                
                # Compute intersection incrementally with pruning
                for subset in selected_subsets[1:]:
                    current_intersection = current_intersection.intersection(subset)
                    
                    # Early termination if intersection becomes too small
                    if len(current_intersection) < min_threshold:
                        choices_pruned += 1
                        break
                else:
                    # Only process if we didn't break early
                    score = len(current_intersection)
                    
                    # Only consider adding if score is promising
                    if score >= min_threshold:
                        item = (score, group_indices, sorted(current_intersection))
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                            # Update threshold when heap becomes full
                            if len(heap) == top_k:
                                min_threshold = heap[0][0]
                        else:
                            if item > heap[0]:
                                heapq.heappushpop(heap, item)
                                # Update threshold with new minimum
                                min_threshold = heap[0][0]
                    choices_processed += 1
            
            t3 = time.perf_counter()
            max_choices_time = max(max_choices_time, t3 - t2)
            
            if choices_pruned > 0:
                stats["pruned_choices"] += choices_pruned
                if choices_pruned == num_choices:
                    stats["pruned_combinations"] += 1
                    
        t1 = time.perf_counter()
        stats["combinations_time_s"] = t1 - t0
        stats["max_choices_time"] = max_choices_time
        stats["average_choices_per_combination"] = (
            stats["total_choices"] / stats["combinations"]
        )
        if "pruned_choices" in stats:
            stats["pruning_rate"] = stats["pruned_choices"] / stats["total_choices"]
        else:
            stats["pruning_rate"] = 0.0

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
        """
        Pruning-optimized version of max_n_subset_intersection_brute_force.
        
        This version uses the pruning-enabled intersection function.
        """
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

        # Use the pruning-optimized intersection function
        top_intersections = MatrixAppPruning.max_n_representative_intersection_with_pruning(
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