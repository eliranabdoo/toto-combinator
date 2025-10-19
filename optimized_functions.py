import numpy as np
import heapq
from itertools import combinations, groupby, product
from typing import List, Tuple
from core_functions import CalculationParams, ProcessingMode

def max_n_subset_intersection_brute_force_with_pruning(matrix: np.ndarray, params: CalculationParams) -> List[Tuple[list[int], list[int], list[Tuple[int, int]]]]:
    """
    Optimized version with intersection pruning and early termination.
    
    Key optimizations:
    1. Incremental intersection with early termination
    2. Dynamic threshold updates based on current heap minimum
    3. Pre-compute set sizes for better pruning decisions
    """
    per_col_subsets = []

    # Build subsets (same as original)
    for col in matrix.T:
        index_and_elem_pairs = sorted(list(enumerate(col)), key=lambda p: p[1])
        groups = [list(g) for _, g in groupby(index_and_elem_pairs, key=lambda p: p[1])]
        if params.relaxation > 0:
            relaxed_groups = []
            for group_index, group in enumerate(groups):
                relaxed_group = group.copy()
                relaxed_groups.append(relaxed_group)
                if len(group) == 0:
                    continue
                group_value = group[0][1]
                for added_group in groups[group_index + 1:]:
                    if (added_group[0][1] - group_value) > params.relaxation:
                        break
                    relaxed_group.extend(added_group.copy())
            groups = relaxed_groups
        subsets = [set([p[0] for p in group]) for group in groups]
        per_col_subsets.append(subsets)

    # Pre-compute set sizes for better pruning decisions
    subset_sizes = []
    for group in per_col_subsets:
        group_sizes = [len(subset) for subset in group]
        subset_sizes.append(group_sizes)

    # Use optimized intersection function with pruning
    top_intersections = max_n_representative_intersection_with_pruning(
        per_col_subsets, subset_sizes, n=params.n, top_k=params.top_k
    )

    # Convert results (same as original)
    res = []
    for intersection in top_intersections:
        chosen_columns, chosen_rows = intersection
        chosen_columns_ranges = []

        for col in chosen_columns:
            view = matrix[chosen_rows, col]
            col_range = (view.min(), view.max())
            chosen_columns_ranges.append(col_range)
            assert col_range[1] - col_range[0] <= params.relaxation, "invalid relaxation"
        res.append((chosen_columns, chosen_rows, chosen_columns_ranges))

    return res

def max_n_representative_intersection_with_pruning(groups, subset_sizes, n, top_k=1):
    """
    Optimized intersection function with pruning.
    
    Key optimizations:
    1. Incremental intersection with early termination
    2. Dynamic threshold updates
    3. Size-based pruning before expensive intersection
    """
    heap = []
    min_threshold = 0  # Start with no pruning

    # All combinations of n groups out of N
    for group_indices in combinations(range(len(groups)), n):
        selected_groups = [groups[i] for i in group_indices]
        selected_sizes = [subset_sizes[i] for i in group_indices]
        
        # All ways to pick one subset from each selected group
        for choice in product(*[range(len(g)) for g in selected_groups]):
            # Quick size-based pruning: if any subset is too small, skip
            if any(selected_sizes[i][choice[i]] < min_threshold for i in range(n)):
                continue
                
            selected_subsets = [selected_groups[i][choice[i]] for i in range(n)]
            
            # Incremental intersection with early termination
            current_intersection = compute_intersection_with_pruning(
                selected_subsets, min_threshold
            )
            
            if current_intersection is None:  # Pruned
                continue
                
            score = len(current_intersection)
            
            # Use a min-heap of size top_k
            item = (score, group_indices, sorted(current_intersection))
            if len(heap) < top_k:
                heapq.heappush(heap, item)
                # Update threshold when heap is full
                if len(heap) == top_k:
                    min_threshold = heap[0][0]
            else:
                if item > heap[0]:
                    heapq.heappushpop(heap, item)
                    # Update threshold after adding better item
                    min_threshold = heap[0][0]

    # Sort results by intersection size descending, then by group indices
    results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
    return [(list(group_indices), intersection) for _, group_indices, intersection in results]

def compute_intersection_with_pruning(selected_subsets, min_threshold):
    """
    Compute intersection with early termination pruning.
    
    Returns None if intersection becomes too small, otherwise returns the intersection set.
    """
    current_intersection = selected_subsets[0]
    
    for i, subset in enumerate(selected_subsets[1:], 1):
        current_intersection = current_intersection.intersection(subset)
        
        # Early termination: if intersection too small, skip this combination
        if len(current_intersection) < min_threshold:
            return None  # Prune this branch
    
    return current_intersection

def max_n_subset_intersection_brute_force_with_parallelization(matrix: np.ndarray, params: CalculationParams) -> List[Tuple[list[int], list[int], list[Tuple[int, int]]]]:
    """
    Optimized version with smart parallelization strategy.
    
    Key optimizations:
    1. Scale-based parallelization decision (column-level vs choice-level)
    2. Column-level parallelization when C^n >> ∏(M_i)
    3. Choice-level parallelization when ∏(M_i) >> C^n
    """
    per_col_subsets = []

    # Build subsets (same as original)
    for col in matrix.T:
        index_and_elem_pairs = sorted(list(enumerate(col)), key=lambda p: p[1])
        groups = [list(g) for _, g in groupby(index_and_elem_pairs, key=lambda p: p[1])]
        if params.relaxation > 0:
            relaxed_groups = []
            for group_index, group in enumerate(groups):
                relaxed_group = group.copy()
                relaxed_groups.append(relaxed_group)
                if len(group) == 0:
                    continue
                group_value = group[0][1]
                for added_group in groups[group_index + 1:]:
                    if (added_group[0][1] - group_value) > params.relaxation:
                        break
                    relaxed_group.extend(added_group.copy())
            groups = relaxed_groups
        subsets = [set([p[0] for p in group]) for group in groups]
        per_col_subsets.append(subsets)

    # Choose parallelization strategy based on scale analysis
    strategy = choose_parallelization_strategy(per_col_subsets, params.n)
    
    if strategy == "column_level":
        top_intersections = max_n_representative_intersection_parallel_columns(
            per_col_subsets, n=params.n, top_k=params.top_k
        )
    else:  # choice_level
        top_intersections = max_n_representative_intersection_parallel_choices(
            per_col_subsets, n=params.n, top_k=params.top_k
        )

    # Convert results (same as original)
    res = []
    for intersection in top_intersections:
        chosen_columns, chosen_rows = intersection
        chosen_columns_ranges = []

        for col in chosen_columns:
            view = matrix[chosen_rows, col]
            col_range = (view.min(), view.max())
            chosen_columns_ranges.append(col_range)
            assert col_range[1] - col_range[0] <= params.relaxation, "invalid relaxation"
        res.append((chosen_columns, chosen_rows, chosen_columns_ranges))

    return res

def choose_parallelization_strategy(groups, n):
    """
    Choose optimal parallelization strategy based on scale analysis.
    
    Returns "column_level" if C^n >> ∏(M_i), "choice_level" otherwise.
    """
    import math
    
    num_columns = len(groups)
    num_column_combinations = math.comb(num_columns, n)
    
    # Calculate max choices per combination
    max_choices_per_combination = 0
    for combo in combinations(range(num_columns), n):
        choices = math.prod(len(groups[i]) for i in combo)
        max_choices_per_combination = max(max_choices_per_combination, choices)
    
    # If column combinations dominate, use column-level parallelization
    if num_column_combinations > max_choices_per_combination * 10:
        return "column_level"
    else:
        return "choice_level"

def max_n_representative_intersection_parallel_columns(groups, n, top_k=1):
    """
    Column-level parallelization: distribute column combinations across processes.
    """
    import multiprocessing as mp
    from functools import partial
    
    # Get all column combinations
    all_combinations = list(combinations(range(len(groups)), n))
    
    # Use available CPU cores
    num_processes = min(mp.cpu_count(), len(all_combinations))
    
    if num_processes <= 1:
        # Fall back to sequential processing
        return max_n_representative_intersection_sequential(groups, n, top_k)
    
    # Distribute combinations across processes
    chunk_size = max(1, len(all_combinations) // num_processes)
    chunks = [all_combinations[i:i+chunk_size] for i in range(0, len(all_combinations), chunk_size)]
    
    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        worker_func = partial(process_column_combination_chunk, groups, top_k)
        chunk_results = pool.map(worker_func, chunks)
    
    # Merge results from all chunks
    all_heaps = [result for result in chunk_results if result]
    if not all_heaps:
        return []
    
    # Merge heaps and keep top_k
    merged_heap = []
    for heap in all_heaps:
        for item in heap:
            if len(merged_heap) < top_k:
                heapq.heappush(merged_heap, item)
            else:
                if item > merged_heap[0]:
                    heapq.heappushpop(merged_heap, item)
    
    # Sort results
    results = sorted(merged_heap, key=lambda x: (x[0], x[1]), reverse=True)
    return [(list(group_indices), intersection) for _, group_indices, intersection in results]

def process_column_combination_chunk(groups, top_k, combination_chunk):
    """
    Process a chunk of column combinations (worker function for parallel processing).
    """
    heap = []
    
    for group_indices in combination_chunk:
        selected_groups = [groups[i] for i in group_indices]
        
        # All ways to pick one subset from each selected group
        for choice in product(*[range(len(g)) for g in selected_groups]):
            selected_subsets = [selected_groups[i][choice[i]] for i in range(len(group_indices))]
            current_intersection = selected_subsets[0]
            for subset in selected_subsets[1:]:
                current_intersection = current_intersection.intersection(subset)
            score = len(current_intersection)
            
            item = (score, group_indices, sorted(current_intersection))
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if item > heap[0]:
                    heapq.heappushpop(heap, item)
    
    return heap

def max_n_representative_intersection_parallel_choices(groups, n, top_k=1):
    """
    Choice-level parallelization: distribute choice combinations across processes.
    """
    import multiprocessing as mp
    from functools import partial
    
    # Get all column combinations first
    all_combinations = list(combinations(range(len(groups)), n))
    
    # For each column combination, get all choice combinations
    all_work_items = []
    for group_indices in all_combinations:
        selected_groups = [groups[i] for i in group_indices]
        choices = list(product(*[range(len(g)) for g in selected_groups]))
        all_work_items.append((group_indices, selected_groups, choices))
    
    # Use available CPU cores
    num_processes = min(mp.cpu_count(), len(all_work_items))
    
    if num_processes <= 1:
        # Fall back to sequential processing
        return max_n_representative_intersection_sequential(groups, n, top_k)
    
    # Process work items in parallel
    with mp.Pool(processes=num_processes) as pool:
        worker_func = partial(process_choice_combinations, top_k)
        chunk_results = pool.map(worker_func, all_work_items)
    
    # Merge results from all chunks
    all_heaps = [result for result in chunk_results if result]
    if not all_heaps:
        return []
    
    # Merge heaps and keep top_k
    merged_heap = []
    for heap in all_heaps:
        for item in heap:
            if len(merged_heap) < top_k:
                heapq.heappush(merged_heap, item)
            else:
                if item > merged_heap[0]:
                    heapq.heappushpop(merged_heap, item)
    
    # Sort results
    results = sorted(merged_heap, key=lambda x: (x[0], x[1]), reverse=True)
    return [(list(group_indices), intersection) for _, group_indices, intersection in results]

def process_choice_combinations(top_k, work_item):
    """
    Process choice combinations for a single column combination (worker function).
    """
    group_indices, selected_groups, choices = work_item
    heap = []
    
    for choice in choices:
        selected_subsets = [selected_groups[i][choice[i]] for i in range(len(group_indices))]
        current_intersection = selected_subsets[0]
        for subset in selected_subsets[1:]:
            current_intersection = current_intersection.intersection(subset)
        score = len(current_intersection)
        
        item = (score, group_indices, sorted(current_intersection))
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        else:
            if item > heap[0]:
                heapq.heappushpop(heap, item)
    
    return heap

def max_n_representative_intersection_sequential(groups, n, top_k=1):
    """
    Sequential fallback implementation (same as original).
    """
    heap = []

    # All combinations of n groups out of N
    for group_indices in combinations(range(len(groups)), n):
        selected_groups = [groups[i] for i in group_indices]
        # All ways to pick one subset from each selected group
        for choice in product(*[range(len(g)) for g in selected_groups]):
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

    # Sort results by intersection size descending, then by group indices
    results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
    return [(list(group_indices), intersection) for _, group_indices, intersection in results]