#!/usr/bin/env python3
"""
Test data generator for optimization validation
Creates test matrices and ground truth data for the max_n_subset_intersection_brute_force function
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from app import MatrixApp, CalculationParams, ProcessingMode

def create_test_matrix_131():
    """Create a 13x1 matrix similar to the 131.xlsx test case"""
    # Create a matrix with some repeated values to create interesting subsets
    matrix = np.array([
        [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6], [7]
    ])
    return matrix

def create_test_matrix_243():
    """Create a 24x3 matrix similar to the 243.xlsx test case"""
    # Create a matrix with more complex patterns
    np.random.seed(42)  # For reproducibility
    matrix = np.random.randint(1, 6, size=(24, 3))
    return matrix

def create_test_matrices():
    """Create and save test matrices"""
    os.makedirs("tests/assets", exist_ok=True)
    
    # Create 131.xlsx
    matrix_131 = create_test_matrix_131()
    df_131 = pd.DataFrame(matrix_131)
    df_131.to_excel("tests/assets/131.xlsx", index=False, header=False)
    
    # Create 243.xlsx  
    matrix_243 = create_test_matrix_243()
    df_243 = pd.DataFrame(matrix_243)
    df_243.to_excel("tests/assets/243.xlsx", index=False, header=False)
    
    return matrix_131, matrix_243

def generate_ground_truth(matrix, matrix_name, n, relaxation, top_k):
    """Generate ground truth results for a test case"""
    params = CalculationParams(
        n=n,
        mode=ProcessingMode.SEPARATE,
        relaxation=relaxation,
        top_k=top_k
    )
    
    # Run the original function to get ground truth
    results = MatrixApp.max_n_subset_intersection_brute_force(matrix, params)
    
    # Convert to the format expected by test cases
    test_case = {
        'matrix_file_path': f"tests/assets/{matrix_name}.xlsx",
        'params': params,
        'expected_results': results
    }
    
    return test_case

def create_ground_truth_files():
    """Create all ground truth files for testing"""
    os.makedirs("tests/assets/ground_truths", exist_ok=True)
    
    # Create test matrices
    matrix_131, matrix_243 = create_test_matrices()
    
    # Test cases to generate
    test_cases = [
        (matrix_131, "131", 3, 0, 1),
        (matrix_131, "131", 4, 0, 2),
        (matrix_243, "243", 3, 1, 1),
        (matrix_243, "243", 4, 1, 2),
    ]
    
    for matrix, name, n, relaxation, top_k in test_cases:
        test_case = generate_ground_truth(matrix, name, n, relaxation, top_k)
        
        filename = f"tests/assets/ground_truths/{name}_{n}_ProcessingMode.SEPARATE_{relaxation}_{top_k}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(test_case, f)
        print(f"Created ground truth: {filename}")

if __name__ == "__main__":
    create_ground_truth_files()
    print("Test data generation complete!")