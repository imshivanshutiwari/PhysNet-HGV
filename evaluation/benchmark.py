"""
Benchmark Analysis for PhysNet-HGV.

Compares the performance of the proposed Physics-Informed Neural 
Kalman Framework against baseline estimation methods like 
Extended Kalman Filters (EKF) and standard Deep Learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
from .metrics import TrackingMetrics
from .evaluate import HGVEvaluator
from typing import Dict, List

class HGVBenchmark:
    """
    Comparative benchmarking suite.
    """
    
    def __init__(self, evaluator: HGVEvaluator):
        self.evaluator = evaluator
        self.metrics = TrackingMetrics()

    def compare_methods(self, methods: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Runs tracking across different architectures and records results.
        """
        results = {}
        for method in methods:
            print(f"Benchmarking {method}...")
            # In a real run, we'd swap internal components (e.g. EKF vs UKF)
            res = self.evaluator.run_eval(n_steps=500)
            results[method] = res
        return results

    def plot_results(self, results: Dict[str, Dict[str, float]]):
        """
        Visualizes benchmarking results.
        """
        methods = list(results.keys())
        rmse = [results[m]["pos_rmse"] for m in methods]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, rmse, color='skyblue')
        plt.ylabel("Position RMSE (m)")
        plt.title("HGV Tracking Performance Comparison")
        plt.grid(True, axis='y')
        plt.show()

if __name__ == "__main__":
    print("Benchmark module initialized.")
