"""
Test the pruning optimization against the baseline
"""

import json
import pickle
from optimization_testing import run_optimization_validation_pipeline, collect_baseline_metrics
from optimization_pruning import MatrixAppPruning


def main():
    # Load baseline metrics if available, otherwise collect them
    try:
        with open("baseline_metrics.json", "r") as f:
            baseline_json = json.load(f)
            print("Loaded existing baseline metrics from baseline_metrics.json")
            
            # Since we stored simplified JSON, we need to recreate the full baseline
            # by re-running the tests to get the results
            baseline_results = collect_baseline_metrics()
    except FileNotFoundError:
        print("No existing baseline found, collecting baseline metrics...")
        baseline_results = collect_baseline_metrics()
    
    # Test the pruning optimization
    print("\n" + "="*80)
    print("TESTING PRUNING OPTIMIZATION")
    print("="*80)
    
    results = run_optimization_validation_pipeline(
        optimization_name="Pruning Optimization",
        optimized_function=MatrixAppPruning.max_n_subset_intersection_brute_force,
        baseline_results=baseline_results
    )
    
    # Print detailed statistics about pruning
    print("\n" + "="*80)
    print("PRUNING STATISTICS")
    print("="*80)
    
    for case_path, perf_data in results['optimization']['performance_gains'].items():
        case_name = perf_data['case_name']
        stats = perf_data.get('stats', {})
        
        pruning_rate = stats.get('pruning_rate', 0) * 100 if stats else 0
        pruned_choices = stats.get('pruned_choices', 0) if stats else 0
        total_choices = stats.get('total_choices', 0) if stats else 0
        
        print(f"\n{case_name}:")
        print(f"  Pruning rate: {pruning_rate:.1f}%")
        print(f"  Pruned choices: {pruned_choices:,} / {total_choices:,}")
        print(f"  Speedup: {perf_data['speedup']:.2f}x")
    
    return results


if __name__ == "__main__":
    main()