import logging
import numpy as np
import pytest
from app import MatrixApp, CalculationParams, ProcessingMode


EXCEL_FILE_PATH = "tests/assets/131.xlsx"
logging.basicConfig(level=logging.DEBUG)

# Placeholder for testing Excel file loading
def test_load_excel():
    matrix = MatrixApp.load_matrix_from_excel(EXCEL_FILE_PATH, columns_count=12)
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
    matrix = np.array([
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
    ])
    params = CalculationParams(
        n=3,
        mode=ProcessingMode.SEPARATE,
        ranges=[(2, 3), (6, 8), (10, 12)]
    )
    filtered_columns = MatrixApp.filter_columns_by_ranges(matrix, params.ranges)
    assert filtered_columns == [{1,2}, {1,2,3}, {1,2,3}]

def test_max_k_subset_intersection_brute_force():
    # Subsets as sets of row indices
    subsets = [
        {1,2},
        {1,2,3},
        {1,2,3}
    ]
    k = 3
    max_size, best = MatrixApp.max_n_subset_intersection_brute_force(subsets, k)
    assert max_size == 2
    assert list(best[0]) == [0,1,2] or list(best[0]) == [0,2,1] or list(best[0]) == [1,0,2]  # any order
    assert best[1] == {1,2} 

def test_max_n_subset_intersection_brute_force_with_243():
    """Test max_n_subset_intersection_brute_force with 243.xlsx data"""
    # Load the matrix from 243.xlsx
    matrix = MatrixApp.load_matrix_from_excel("tests/assets/243.xlsx")
    
    # Set up parameters: n=4, no relaxation, top_k=2
    params = CalculationParams(
        n=3,
        mode=ProcessingMode.SEPARATE,
        relaxation=0,
        top_k=2
    )
    
    # Run the calculation
    results = MatrixApp.max_n_subset_intersection_brute_force(matrix, params)
    
    # Verify results
    assert len(results) == 2  # Should return top_k=2 results