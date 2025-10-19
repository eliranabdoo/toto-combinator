from dataclasses import dataclass
import os
import pickle
from app import CalculationResult, MatrixApp, CalculationParams, ProcessingMode
from tqdm import tqdm

@dataclass
class MatrixAppTestCase:
    matrix_file_path: str
    params: CalculationParams
    expected_results: list[CalculationResult]


def generate_test_cases(path_to_params_sets: dict[str, list[CalculationParams]], pickles_dst_dir: str):
    for path, params_set in tqdm(path_to_params_sets.items(), desc="generating test cases"):  # pyright: ignore[reportUndefinedVariable]
        matrix = MatrixApp.load_matrix_from_excel(path)
        path_id = os.path.basename(path).split(".")[0]
        for params in params_set:
            params_id = "_".join(str(params.__getattribute__(f)) for f in params.__dataclass_fields__)
            result_id = f"{path_id}_{params_id}"
            result = MatrixApp.run_calculation(matrix=matrix, params=params)
            test_case = MatrixAppTestCase(
                matrix_file_path=path,
                params=params,
                expected_results=result
            )
            result_path = os.path.join(pickles_dst_dir, f"{result_id}.pickle")
            pickle.dump(test_case, open(result_path, "wb"))


def main():
    params_set = [
        CalculationParams(mode=ProcessingMode.SEPARATE, n=3, relaxation=0, top_k=1),
        CalculationParams(mode=ProcessingMode.SEPARATE, n=4, relaxation=0, top_k=2),
        CalculationParams(mode=ProcessingMode.SEPARATE, n=3, relaxation=1, top_k=2),
        CalculationParams(mode=ProcessingMode.SEPARATE, n=4, relaxation=1, top_k=1),
        CalculationParams(mode=ProcessingMode.AGGREGATIVE, n=3, relaxation=0, top_k=2),
        CalculationParams(mode=ProcessingMode.AGGREGATIVE, n=4, relaxation=0, top_k=1),
    ]

    path_to_params_sets = {
        "tests/assets/131.xlsx": params_set,
        "tests/assets/243.xlsx": params_set,
    }

    generate_test_cases(
        path_to_params_sets=path_to_params_sets,
        pickles_dst_dir="tests/assets/test_cases"
    )

if __name__ == "__main__":
    main()
