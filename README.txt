ISEF Dry Lab Package (Gompertz Growth Simulation)

Files included:
  - cell_growth.py
  - parameters.csv
  - scenarios.csv
  - effects.csv

How to run (macOS / Windows):
  1) Install dependencies:
       pip install pandas numpy
  2) Put all files in the same folder.
  3) Run:
       python cell_growth.py

Outputs generated in the same folder:
  - simulated_curves.csv
  - growth_metrics.csv
  - best_concentrations.csv

Notes for your write-up:
  - Dopamine and acetylcholine are treated as constants at 500 uM across all scenarios.
  - Effect magnitudes are hypothesis-driven rules (not measured data) and should be described as such.
