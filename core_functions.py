import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from itertools import combinations, groupby, product
import heapq

class ProcessingMode(Enum):
    AGGREGATIVE = "סיכומי"
    SEPARATE = "נפרד"

@dataclass
class CalculationParams:
    n: int
    mode: ProcessingMode
    relaxation: int
    top_k: int

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
    def max_n_representative_intersection(groups, n, top_k=1) -> List[Tuple[list[int], list[int]]]:
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

    @staticmethod
    def max_n_subset_intersection_brute_force(matrix: np.ndarray, params: CalculationParams) -> List[Tuple[list[int], list[int], list[Tuple[int, int]]]]:
        per_col_subsets = []

        for col in matrix.T:
            index_and_elem_pairs = sorted(list(enumerate(col)), key=lambda p: p[1])
            groups = [list(g) for _, g in groupby(index_and_elem_pairs, key=lambda p: p[1])]  # [[(0,0), (1,0), (2,0)], [(3,1), (7,1)] ...]
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

        top_intersections = MatrixApp.max_n_representative_intersection(per_col_subsets, n=params.n, top_k=params.top_k)

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