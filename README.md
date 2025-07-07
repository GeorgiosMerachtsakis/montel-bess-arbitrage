
# 🔋 Montel – Battery Arbitrage Strategy

This repository contains a Python implementation for simulating the **bidding strategy of a battery system** in an electricity market. The work was completed as part of the course _31762 – Introduction to Energy Analytics_ at DTU.

---

## 📌 Assignment Overview

The objective of the project is to model the optimal charge/discharge behavior of a battery system participating in a day-ahead electricity market. The optimization considers:

- Battery technical constraints (state-of-charge, capacity, efficiency)
- Forecasted market prices
- Arbitrage opportunities
- Economic performance (profit maximization)

The model was developed in Python using linear programming with the `PuLP` package.

---

## 📁 Files

| File | Description |
|------|-------------|
| `battery_bidding_strategy.py` | Main optimization script |
| `docs/Assignment_BESS.pdf` | Assignment instructions and scenario details |

---

## 🛠 Requirements

- Python 3.7+
- `pulp`
- `pandas`
- `matplotlib` (for optional plotting)

Install dependencies with:

```bash
pip install pulp pandas matplotlib
