# tests/test_eval.py
from src.evaluation.runner import run_eval_suite

# Quick run — first 5 questions only
# Remove limit=5 to run the full 50
summary = run_eval_suite(limit=5, save_results=True)

print(f"\nFinal composite: {summary['overall']['composite']:.0%}")