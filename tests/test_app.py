import logging
import pickle
import numpy as np
import pytest
from data_models import CalculationParams, ProcessingMode
from app import MatrixApp
from generate_test_cases import MatrixAppTestCase


logging.basicConfig(level=logging.DEBUG)


# Placeholder for testing Excel file loading
def test_load_excel():
    excel_file_path = "tests/assets/131.xlsx"
    matrix = MatrixApp.load_matrix_from_excel(excel_file_path, columns_count=12)
    assert matrix is not None
    assert matrix.dtype == np.int32


# Placeholder for testing parameter handling
def test_dummy_parameter():
    pass


# Placeholder for testing calculation logic
def test_calculation():
    pass


# Placeholder for testing output file creation
def test_output_file():
    pass


def test_filtering_by_range():
    # Matrix: 4 rows x 3 columns
    matrix = np.array(
        [
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12],
        ]
    )
    params = CalculationParams(
        n=3, mode=ProcessingMode.SEPARATE, ranges=[(2, 3), (6, 8), (10, 12)]
    )
    filtered_columns = MatrixApp.filter_columns_by_ranges(matrix, params.ranges)
    assert filtered_columns == [{1, 2}, {1, 2, 3}, {1, 2, 3}]


def test_max_k_subset_intersection_brute_force():
    # Subsets as sets of row indices
    subsets = [{1, 2}, {1, 2, 3}, {1, 2, 3}]
    k = 3
    max_size, best = MatrixApp.max_n_subset_intersection_brute_force(subsets, k)
    assert max_size == 2
    assert (
        list(best[0]) == [0, 1, 2]
        or list(best[0]) == [0, 2, 1]
        or list(best[0]) == [1, 0, 2]
    )  # any order
    assert best[1] == {1, 2}


GROUD_TRUTHS_DIR = "tests/assets/ground_truths"


@pytest.mark.parametrize(
    "case_pickle_path",
    # [os.path.join(GROUD_TRUTHS_DIR, p) for p in os.listdir(GROUD_TRUTHS_DIR)],
    ["tests/assets/ground_truths/131_4_ProcessingMode.AGGREGATIVE_0_1.pickle"],
)
def test_a_case(case_pickle_path):
    test_case: MatrixAppTestCase = pickle.load(open(case_pickle_path, "rb"), fix_imports=False)
    matrix = MatrixApp.load_matrix_from_excel(test_case.matrix_file_path)
    res, stats = MatrixApp.run_calculation(        
        matrix=matrix,
        params=test_case.params,
        return_stats=True
    )

    assert all([
        result == desired for result, desired in zip(res, test_case.expected_results)
    ])
