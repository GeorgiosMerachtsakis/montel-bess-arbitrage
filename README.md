bess-arbitrage-model
Battery energy storage system (BESS) arbitrage simulation and optimization in Python.

This repository was developed as part of a university assignment focused on modeling battery storage operation in electricity markets using Python and Pyomo. The model simulates day-ahead bidding of a 1 MW / 2 MWh battery system based on hourly price signals.

## Repository Structure
├── BESS_arbitrage_model.py
├── elspotprices_19to21.csv
├── data_sources/
│   └── Assignment_Description.pdf
└── README.md

## Assignment:Battery Arbitrage Modeling
This assignment explores the economic potential of battery energy storage systems by implementing a simple daily arbitrage strategy. The model determines the optimal charging and discharging profile for a fixed-size battery, using historical spot market prices in Denmark.

## Model Description

- A 1 MW / 2 MWh battery is operated in a single price area (e.g., DK2)

- Charging and discharging decisions are optimized for each day separately

- The battery must return to its original state of charge by the end of each day

- Optional roundtrip efficiency losses are included in the simulation

- Optimization is formulated and solved using Pyomo and the GLPK solver

## Data Sources

All input data and assignment description are located in [`data/`](./data) and [`doc/`](./doc):
- elspotprices_19to21.csv: Hourly day-ahead electricity prices for DK price zones
- Assignment_Description.pdf: Problem statement and task overview

## Requirements

-Python ≥ 3.7
-pandas
-numpy
-matplotlib
-pyomo
-glpk (as external solver)
