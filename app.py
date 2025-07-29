from functools import reduce
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Set, Tuple
from enum import Enum, auto
from itertools import combinations, groupby, product

class ProcessingMode(Enum):
    AGGREGATIVE = "סיכומי"
    SEPARATE = "נפרד"


@dataclass
class CalculationResult:
    chosen_columns_ranges: List[Tuple[int, int]]
    chosen_rows: List[int]
    chosen_columns: List[int]

@dataclass
class CalculationParams:
    n: int
    mode: ProcessingMode
    relaxation: int

class MatrixApp:
    def __init__(self, root, columns_count: int = 12):
        self.root = root
        self.root.title("מחשבון קומבינציות")
        self.root.geometry("500x600")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        self.file_path = None
        self.columns_count = columns_count
        
        # Main container
        main_frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="מחשבון קומבינציות", 
                              font=('Arial', 16, 'bold'), 
                              bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # File selection frame
        file_frame = tk.LabelFrame(main_frame, text="בחירת קובץ", 
                                  font=('Arial', 10, 'bold'),
                                  bg='#f0f0f0', fg='#2c3e50',
                                  padx=15, pady=10)
        file_frame.pack(fill='x', pady=(0, 15))
        
        self.file_label = tk.Label(file_frame, text="לא נבחר קובץ", 
                                  bg='#f0f0f0', fg='#7f8c8d',
                                  font=('Arial', 9))
        self.file_label.pack(pady=(0, 5))
        
        self.browse_button = tk.Button(file_frame, text="בחר קובץ Excel", 
                                      command=self.browse_file,
                                      bg='#3498db', fg='white',
                                      font=('Arial', 9, 'bold'),
                                      relief='flat', padx=15, pady=5)
        self.browse_button.pack()

        # Parameters frame
        params_frame = tk.LabelFrame(main_frame, text="פרמטרים", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50',
                                    padx=15, pady=10)
        params_frame.pack(fill='x', pady=(0, 15))

        # Number of configurations (N)
        n_frame = tk.Frame(params_frame, bg='#f0f0f0')
        n_frame.pack(fill='x', pady=(0, 10))
        
        self.n_label = tk.Label(n_frame, text="מספר טורים רצוי", 
                               bg='#f0f0f0', font=('Arial', 9))
        self.n_label.pack(anchor='w')
        
        self.n_var = tk.IntVar(value=2)
        self.n_entry = tk.Entry(n_frame, textvariable=self.n_var, 
                               font=('Arial', 9), width=10)
        self.n_entry.pack(anchor='w', pady=(5, 0))
        # self.n_entry.bind('<KeyRelease>', self.on_n_change)

        # Relaxation number
        relax_frame = tk.Frame(params_frame, bg='#f0f0f0')
        relax_frame.pack(fill='x', pady=(0, 10))
        self.relax_label = tk.Label(relax_frame, text="מספר קפיצות מקסימלי", bg='#f0f0f0', font=('Arial', 9))
        self.relax_label.pack(anchor='w')
        self.relax_var = tk.IntVar(value=0)
        self.relax_entry = tk.Entry(relax_frame, textvariable=self.relax_var, font=('Arial', 9), width=10)
        self.relax_entry.pack(anchor='w', pady=(5, 0))

        # Processing mode
        mode_frame = tk.Frame(params_frame, bg='#f0f0f0')
        mode_frame.pack(fill='x', pady=(0, 10))
        
        self.mode_label = tk.Label(mode_frame, text="מצב עיבוד", 
                                  bg='#f0f0f0', font=('Arial', 9))
        self.mode_label.pack(anchor='w')
        
        self.mode_var = tk.StringVar(value=ProcessingMode.SEPARATE.value)
        self.agg_radio = tk.Radiobutton(mode_frame, text="סיכומי", 
                                       variable=self.mode_var, 
                                       value=ProcessingMode.AGGREGATIVE.value, 
                                       bg='#f0f0f0', font=('Arial', 9))
        self.sep_radio = tk.Radiobutton(mode_frame, text="נפרד", 
                                       variable=self.mode_var, 
                                       value=ProcessingMode.SEPARATE.value, 
                                       bg='#f0f0f0', font=('Arial', 9))
        self.agg_radio.pack(anchor='w', pady=(5, 0))
        self.sep_radio.pack(anchor='w')

        # Run button
        self.run_button = tk.Button(main_frame, text="הרץ חישוב", 
                                   command=self.run,
                                   bg='#27ae60', fg='white',
                                   font=('Arial', 11, 'bold'),
                                   relief='flat', padx=30, pady=10)
        self.run_button.pack(pady=10)

    def browse_file(self):
        filetypes = [("Excel files", "*.xlsx *.xls")]
        path = filedialog.askopenfilename(title="Select Excel File", filetypes=filetypes)
        if path:
            self.file_path = path
            self.file_label.config(text=os.path.basename(path), fg='#2c3e50')

    def on_n_change(self, event=None):
        try:
            n = int(self.n_entry.get())
            if n < 1:
                n = 1
        except ValueError:
            n = 1
        self.n_var.set(n)

    @staticmethod
    def load_matrix_from_excel(file_path, columns_count: int) -> np.ndarray: 
        df = pd.read_excel(file_path, header=None)
        df = df.iloc[:, :columns_count].dropna().astype(int)
        return df.values

    @staticmethod
    def filter_columns_by_ranges(matrix, ranges):
        filtered_columns = []
        for col_idx, (min_val, max_val) in enumerate(ranges):
            col = matrix[:, col_idx]
            indices = {row_idx for row_idx, x in enumerate(col) if min_val <= x <= max_val}
            filtered_columns.append(indices)
        return filtered_columns
    
    @staticmethod
    def max_k_representative_intersection(groups, k) -> Tuple[list[int], list[int]]:
        max_intersection_size = -1
        best_intersection = None
        best_selected_subsets = None
        best_group_indices = None

        # All combinations of k groups out of N
        for group_indices in combinations(range(len(groups)), k):
            selected_groups = [groups[i] for i in group_indices]
            # All ways to pick one subset from each selected group
            for choice in product(*[range(len(g)) for g in selected_groups]):
                selected_subsets = [selected_groups[i][choice[i]] for i in range(k)]
                current_intersection = reduce(set.intersection, selected_subsets[1:], selected_subsets[0])
                if len(current_intersection) > max_intersection_size:
                    max_intersection_size = len(current_intersection)
                    best_intersection = current_intersection
                    best_selected_subsets = selected_subsets
                    best_group_indices = group_indices
                    
        return sorted(best_group_indices), sorted(best_intersection)

    @staticmethod
    def max_k_subset_intersection_brute_force(matrix: np.ndarray, params: CalculationParams) -> Tuple[list[int], list[int], list[Tuple[int, int]]]:
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
        
        chosen_columns, chosen_rows = MatrixApp.max_k_representative_intersection(per_col_subsets, k=params.n)

        chosen_columns_ranges = []

        for col in chosen_columns:
            view = matrix[chosen_rows, col]
            col_range = (view.min(), view.max())
            chosen_columns_ranges.append(col_range)
            assert col_range[1] - col_range[0] <= params.relaxation, "invalid relaxation"
        
        return chosen_columns, chosen_rows, chosen_columns_ranges

    @staticmethod
    def max_k_subset_sums_brute_force(matrix: np.ndarray, params: CalculationParams) -> Tuple[list[int], list[int], list[Tuple[int, int]]]:
        max_sum_count = -1
        chosen_columns = None
        chosen_rows = None
        chosen_rows_range = None

        if params.relaxation > 0:
            raise NotImplementedError("We don't suppor relaxation for aggregative mode")
        k = params.n
        for combo_indices in combinations(range(matrix.shape[1]), k):
            sums: np.ndarray = matrix[:, combo_indices].sum(axis=1)
            counts = np.bincount(sums)
            max_i = np.argmax(counts)
            max_s = sums[max_i]
            max_count = counts[max_i]

            if max_count > max_sum_count:
                max_sum_count = max_count
                chosen_columns = combo_indices
                chosen_rows = sorted(np.nonzero((sums == max_s)))
                chosen_rows_range = [(max_s, max_s)]
        
        return chosen_columns, chosen_rows, chosen_rows_range
        



            

           
            
        return chosen_columns, chosen_rows, summation_range

    def run_calculation(self, matrix, params: CalculationParams) -> CalculationResult:
        if params.mode == ProcessingMode.SEPARATE:
            chosen_columns, chosen_rows, chosen_columns_ranges = self.max_k_subset_intersection_brute_force(matrix, params)
            return CalculationResult(chosen_columns=chosen_columns,
                                     chosen_columns_ranges=chosen_columns_ranges,
                                     chosen_rows=chosen_rows)
        elif params.mode == ProcessingMode.AGGREGATIVE:
            chosen_columns, chosen_rows, chosen_columns_ranges = self.max_k_subset_sums_brute_force(matrix, params)
            return CalculationResult(chosen_columns=chosen_columns,
                                     chosen_columns_ranges=chosen_columns_ranges,
                                     chosen_rows=chosen_rows)

    def run(self):
        if not self.file_path:
            messagebox.showerror("Error", "בחר קובץ אקסל")
            return
        try:
            matrix = self.load_matrix_from_excel(self.file_path, columns_count=self.columns_count)
            n = self.n_var.get()
            mode_str = self.mode_var.get()
            try:
                mode = ProcessingMode(mode_str)
            except ValueError:
                messagebox.showerror("Error", f"Invalid processing mode: {mode_str}")
                return
            relaxation = self.relax_var.get()
            params = CalculationParams(n=n, mode=mode, relaxation=relaxation)
            result = self.run_calculation(matrix, params)

            base_input_path, extension = os.path.splitext(self.file_path)
            from datetime import datetime
            time_addition = datetime.strftime(datetime.now(), format="%Y%m%d_%H%M%S")
            base_output_path = base_input_path + f"_output_{time_addition}"
            default_output_path = base_output_path + ".txt"
            output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="שמירת תוצאה", initialfile=default_output_path)

            chosen_fixtures = [e+1 for e in result.chosen_rows]
            chosen_configuratios = [c+1 for c in result.chosen_columns]
            chosen_ranges_strs = "/".join([f"{p[0]}-{p[1]}" if len(set(p)) > 1 else f"{p[0]}" for p in result.chosen_columns_ranges])
            if output_path:
                with open(output_path, 'w', encoding="utf-8") as f:
                    f.write(f"כמות עמודות: {params.n}\n")
                    f.write(f"מצב חישוב: {params.mode.value}\n")
                    f.write(f"מספר קפיצות מקסימלי: {params.relaxation}\n")
                    f.write(f"כמות מחזורים: {len(chosen_fixtures)}\n")
                    f.write(f"עמודות נבחרות: {chosen_configuratios}\n")
                    f.write(f"מחזורים נבחרים: {chosen_fixtures}\n")
                    if params.mode == ProcessingMode.SEPARATE:
                        f.write(f"טווחי פגיעה לכל עמודה: {chosen_ranges_strs}\n")    
                    elif params.mode == ProcessingMode.AGGREGATIVE:
                        assert len(chosen_ranges_strs) == 1
                        f.write(f"טווח סיכומי: {chosen_ranges_strs}\n")
                messagebox.showinfo("Success", f"Result saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {e}")

def main():
    root = tk.Tk()
    app = MatrixApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()