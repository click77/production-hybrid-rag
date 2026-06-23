# tests/test_comparison.py
from src.evaluation.compare_strategies import run_strategy_comparison

# Quick test — 3 questions, straightforward only
# Remove limit and categories to run the full 50-question suite
summary = run_strategy_comparison(
    strategies=["fixed", "recursive", "semantic"],
#    limit=3,
#    categories=["straightforward"],
)