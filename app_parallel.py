"""
Parallel implementation for max_n_subset_intersection_brute_force

This module implements parallelization for the brute force algorithm.
"""

import math
import time
import heapq
from typing import Dict, List, Optional, Tuple
from itertools import combinations, groupby, product
from multiprocessing import Pool, Manager
import numpy as np

from data_models import CalculationParams, CalculationResult, ProcessingMode


def process_column_combinations_chunk(args):
    """
    Worker function to process a chunk of column combinations
    
    Args:
        args: Tuple containing (chunk_of_combinations, groups, n, top_k, enable_pruning)
    
    Returns:
        List of (score, group_indices, intersection) tuples
    """
    chunk, groups, n, top_k, enable_pruning = args
    
    local_heap = []
    min_threshold = 0 if enable_pruning else -1
    
    stats = {
        "total_choices": 0,
        "pruned_choices": 0
    }
    
    for group_indices in chunk:
        selected_groups = [groups[i] for i in group_indices]
        
        num_choices = math.prod(len(g) for g in selected_groups)
        stats["total_choices"] += num_choices
        choices_pruned = 0
        
        for choice in product(*[range(len(g)) for g in selected_groups]):
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
                
                # Early termination if intersection becomes too small
                if enable_pruning and len(current_intersection) < min_threshold:
                    choices_pruned += 1
                    stats["pruned_choices"] += 1
                    pruned = True
                    break
            
            if pruned:
                continue
            
            score = len(current_intersection)
            
            # Only consider adding if score is good enough
            if not enable_pruning or score >= min_threshold:
                item = (score, group_indices, sorted(current_intersection))
                if len(local_heap) < top_k:
                    heapq.heappush(local_heap, item)
                    if enable_pruning and len(local_heap) == top_k:
                        min_threshold = local_heap[0][0]
                else:
                    if item > local_heap[0]:
                        heapq.heappushpop(local_heap, item)
                        if enable_pruning:
                            min_threshold = local_heap[0][0]
    
    return local_heap, stats


class MatrixAppParallel:
    @staticmethod
    def max_n_representative_intersection_parallel(
        groups, n, top_k, stats: Optional[dict], enable_pruning: bool = False, num_workers: int = 1
    ) -> List[Tuple[list[int], list[int]]]:
        """
        Parallel version of max_n_representative_intersection
        
        Parallelizes at the column combination level - each worker processes
        a chunk of column combinations independently.
        """
        
        if stats is None:
            stats = {}
        
        # Generate all column combinations
        all_combinations = list(combinations(range(len(groups)), n))
        stats["combinations"] = len(all_combinations)
        
        if num_workers == 1:
            # No parallelization - use the original implementation
            # This is a fallback for single-threaded execution
            heap = []
            stats["total_choices"] = 0
            stats["max_choices_per_combination"] = 0
            stats["pruned_choices"] = 0
            stats["pruned_combinations"] = 0
            max_choices_time = 0.0
            
            min_threshold = 0 if enable_pruning else -1
            
            t0 = time.perf_counter()
            for group_indices in all_combinations:
                selected_groups = [groups[i] for i in group_indices]
                
                num_choices = math.prod(len(g) for g in selected_groups)
                stats["total_choices"] += num_choices
                stats["max_choices_per_combination"] = max(
                    stats["max_choices_per_combination"], num_choices
                )
                
                t2 = time.perf_counter()
                choices_pruned = 0
                
                for choice in product(*[range(len(g)) for g in selected_groups]):
                    selected_subsets = [selected_groups[i][choice[i]] for i in range(n)]
                    current_intersection = selected_subsets[0]
                    
                    if enable_pruning and len(current_intersection) < min_threshold:
                        choices_pruned += 1
                        stats["pruned_choices"] += 1
                        continue
                    
                    pruned = False
                    for subset in selected_subsets[1:]:
                        current_intersection = current_intersection.intersection(subset)
                        
                        if enable_pruning and len(current_intersection) < min_threshold:
                            choices_pruned += 1
                            stats["pruned_choices"] += 1
                            pruned = True
                            break
                    
                    if pruned:
                        continue
                    
                    score = len(current_intersection)
                    
                    if not enable_pruning or score >= min_threshold:
                        item = (score, group_indices, sorted(current_intersection))
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                            if enable_pruning and len(heap) == top_k:
                                min_threshold = heap[0][0]
                        else:
                            if item > heap[0]:
                                heapq.heappushpop(heap, item)
                                if enable_pruning:
                                    min_threshold = heap[0][0]
                
                if choices_pruned == num_choices:
                    stats["pruned_combinations"] += 1
                    
                t3 = time.perf_counter()
                max_choices_time = max(max_choices_time, t3 - t2)
            
            t1 = time.perf_counter()
            stats["combinations_time_s"] = t1 - t0
            stats["max_choices_time"] = max_choices_time
            
        else:
            # Parallel execution
            t0 = time.perf_counter()
            
            # Split combinations into chunks for workers
            chunk_size = max(1, len(all_combinations) // (num_workers * 4))  # More chunks than workers for better load balancing
            chunks = [all_combinations[i:i+chunk_size] for i in range(0, len(all_combinations), chunk_size)]
            
            # Prepare arguments for workers
            worker_args = [(chunk, groups, n, top_k, enable_pruning) for chunk in chunks]
            
            # Process chunks in parallel
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_column_combinations_chunk, worker_args)
            
            # Merge results from all workers
            heap = []
            total_choices = 0
            pruned_choices = 0
            
            for local_heap, local_stats in results:
                total_choices += local_stats["total_choices"]
                pruned_choices += local_stats.get("pruned_choices", 0)
                
                # Merge local heaps into global heap
                for item in local_heap:
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    else:
                        if item > heap[0]:
                            heapq.heappushpop(heap, item)
            
            t1 = time.perf_counter()
            stats["combinations_time_s"] = t1 - t0
            stats["total_choices"] = total_choices
            stats["pruned_choices"] = pruned_choices
            stats["num_workers"] = num_workers
            stats["num_chunks"] = len(chunks)
            
        stats["average_choices_per_combination"] = (
            stats["total_choices"] / stats["combinations"] if stats["combinations"] > 0 else 0
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
        matrix: np.ndarray, params: CalculationParams, stats: Optional[Dict] = None
    ) -> Tuple[List[CalculationResult], Optional[Dict]]:
        """
        Parallel version of max_n_subset_intersection_brute_force
        """
        if stats is None:
            stats = {}
            
        per_col_subsets = []
        
        for col in matrix.T:
            index_and_elem_pairs = sorted(
                list(enumerate(col)), key=lambda p: p[1]
            )
            groups = [
                list(g) for _, g in groupby(index_and_elem_pairs, key=lambda p: p[1])
            ]
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
        
        # Use the parallel intersection function
        top_intersections = MatrixAppParallel.max_n_representative_intersection_parallel(
            per_col_subsets, n=params.n, top_k=params.top_k, stats=stats,
            enable_pruning=params.enable_pruning,
            num_workers=params.num_workers
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
        
        return res, stats
    
    @staticmethod
    def run_calculation(
        matrix, params: CalculationParams, return_stats: bool = False
    ) -> Tuple[List[CalculationResult], Optional[Dict]]:
        """
        Wrapper method to match the interface of MatrixApp.run_calculation
        """
        if params.mode != ProcessingMode.SEPARATE:
            raise NotImplementedError("Parallel optimization only supports SEPARATE mode")
        
        stats = {} if return_stats else None
        res, stats = MatrixAppParallel.max_n_subset_intersection_brute_force(matrix, params, stats)
        return res, stats