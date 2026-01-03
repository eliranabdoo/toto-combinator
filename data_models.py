from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class ProcessingMode(str, Enum):
    AGGREGATIVE = "סיכומי"
    SEPARATE = "נפרד"

@dataclass
class CalculationParams:
    n: int
    mode: ProcessingMode
    relaxation: int
    top_k: int
    enable_pruning: bool = False  # Flag to enable pruning optimization
    num_workers: int = 1  # Number of parallel workers (1 = no parallelization)


@dataclass
class TotoResult:
    turim: List[int]
    mahzorim: List[int]
    tvachim: List[str]
    sikum: Optional[int]


@dataclass
class CalculationResult:
    chosen_columns_ranges: List[Tuple[int, int]]
    chosen_rows: List[int]
    chosen_columns: List[int]

@dataclass
class MatrixAppTestCase:
    matrix_file_path: str
    params: CalculationParams
    expected_results: List[CalculationResult]