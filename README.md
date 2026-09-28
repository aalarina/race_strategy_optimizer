# race_strategy_optimizer
A predictive system for pit-stop strategy and tyre wear

```
src/
├── __init__.py
├── data_loader.py    # Тільки збір та чищення даних з FastF1
├── train.py          # Тільки навчання XGBoost моделі
├── engine.py         # 1. Тільки двигун Монте-Карло (run_monte_carlo)
├── optimizer.py      # 2. Тільки вибір тактик (simulate_flexible_stints)
└── analytics.py      # 3. Тільки аналітичні вікна (get_dynamic_window, get_driver_degradation_bias)
```
