# Predict-then-Optimize Approach for Energy Systems

## Description

Diploma thesis titled **"Predict then Optimize Approach for Energy Systems"** by Bispas Dimitrios, at the Department of Electrical & Computer Engineering of the University of Western Macedonia.
 
The objective is the optimal sizing and operation of residential energy systems (fuel cell, photovoltaics, solar water heater, heat pump, natural gas boiler, battery, thermal storage tank) by minimizing the annual cost, via:
- **Prediction** of demand (Random Forest) for electricity, hot water, and space heating
- **Optimization** with MILP (Mixed Integer Linear Programming) using:
  - Commercial solver **Gurobi**
  - Custom **Branch & Bound with Best-First Search** with heuristic and metaheuristic methods (Myopic, VNS)

---

## File Structure

```
.
├── README.md                           # This file
├── THESIS_PREDICT_THEN_OPTIMIZE.pdf    # The thesis text (165 pages)
├── THESIS_PPTX.pdf                     # The thesis PowerPoint presentation  
│
├── LATEX_SOURCES/                      # LaTeX source code
│   ├── thesis.tex                      # Main thesis file
│   ├── presentation.tex                # Presentation (Beamer)
│   ├── appendix_all.tex                # Appendices
│   ├── refs.bib                        # Bibliography (BibTeX)
│   └── images/                         # Images and diagrams
│       Other files:
│       ├── thesis.loa 
│       ├── thesis.lof 
│       ├── thesis.lol
│       └── thesis.lot
│         
│
├── MODEL_SOLVER_CODES/                 # Python code
│   ├── EMS_MODEL.py                    # MILP model – definition of variables, constraints, objective
│   ├── EMS_GUROBI.py                   # Implementation with Pyomo + Gurobi
│   ├── EMS_BB_BFS.py                   # Branch & Bound + Myopic + VNS + solve_instance()
│   ├── PREDICTION.py                   # Demand prediction with Random Forest + Predict-then-Optimize pipeline
│   └── DATA_GENERATOR.ipynb            # Jupyter notebook for data generation (OCHRE simulator)
│
├── OCHRE_DATA/                         # Residential data (OCHRE_DATA)
│   ├── HOUSE1/                         # Test data for House 1
│   ├── HOUSE1.1/                       # Training data for House 1
│   ├── HOUSE1.2/                       # Training data for House 1
│   ├── HOUSE2/                         # Test data for House 2
│   ├── HOUSE2.1/
│   ├── HOUSE2.2/
│   ├── HOUSE3/                         # Test data for House 3
│   ├── HOUSE3.1/
│   └── HOUSE3.2/
│       In each folder:
│       ├── ELECTRICITY_{ATH|THESS}.csv   # Hourly electricity demand (W)
│       ├── DHW_{ATH|THESS}.csv           # Hourly hot water demand (W)
│       ├── SPACE_HEAT_{ATH|THESS}.csv    # Hourly space heating demand (W)
│       ├── SOLAR_DATA_{ATH|THESS}.csv    # Hourly solar irradiance GHI (W/m²)
│       └── TEMPERATURES_{ATH|THESS}.csv  # Ambient and collector temperatures (°C)
│
└── WEATHER_DATA_ENERGY+/                # EPW weather files
    ├── GRC_Athens.167160_IWEC.epw       # Athens weather data
    └── GRC_Thessaloniki.166220_IWEC.epw # Thessaloniki weather data
```

---

## Software Requirements

### 1. Python and Libraries

**Python 3.9+** is required. The main libraries are:

| Library | Usage |
|------------|-------|
| `gurobipy==12.0.0` | Commercial MILP solver |
| `numpy<2.0` | Numerical operations |
| `pandas` | Data management |
| `scikit-learn` | Random Forest Regressor for predictions |
| `pyomo` | Modeling language (alternative to gurobipy) |
| `matplotlib` / `seaborn` | Results visualization |

### 2. Gurobi Optimizer

- **Academic license** 

### 3. Library Installation

```bash
pip install gurobipy==12.0.0
pip install "numpy<2.0"
pip install pandas scikit-learn pyomo matplotlib seaborn
```

### 4. LaTeX (PDF compilation)

**XeLaTeX** is required

---

## Execution Instructions

### Step 1: Data setup

The data are already available in the `OCHRE_DATA/` folder. If you wish to generate them from scratch:

1. Open `code/DATA_GENERATOR.ipynb` in Google Colab
2. Run the cells in order:
   - **Cell 1**: Installation (OCHRE, numpy, gurobipy)
   - **Cell 2**: Copy the OCHRE XML template
   - **Cell 3**: Create XML files for each house/city
   - **Cell 4**: Generate the CSVs
   - **Cell 5**: Generate the CSVs for identical houses

DATA_GENERATOR.ipynb is designed for Google Colab (uses `/content/` paths). For local execution, path modifications will be required.

### Step 2: Solving with Gurobi (Perfect Information)

To run the commercial Gurobi solver:

```bash
python EMS_GUROBI.py --house HOUSE3 --location THESS --data-dir ../data --hours 8760
```

**Parameters**:
- `--house`: House (HOUSE1, HOUSE2, HOUSE3)
- `--location`: City (ATH, THESS)
- `--data-dir`: Path to the data folder
- `--hours`: Time horizon in hours (8760 for one year)

### Step 3: Branch & Bound (Custom Solver)

For the custom Branch & Bound algorithm with Myopic and VNS:

```bash
python EMS_BB_BFS.py
```

**Note**: The file `EMS_BB_BFS.py` in the main block (`if __name__ == "__main__"`) is configured for HOUSE3/THESS. Modify the line:

```python
data = load_data(location="THESS", T=8760, data_directory=os.path.expanduser("~/DB_WORKSPACE/HOUSE3"))
```

Change `data_directory` to point to the correct folder.

### Step 4: Predict-then-Optimize (Prediction + Optimization)

For the full predict-then-optimize pipeline:

```bash
python PREDICTION.py
```

**Flow**:
1. Loads data from the train houses (e.g., HOUSE3.1, HOUSE3.2) and the test house (HOUSE3)
2. Trains Random Forest models for predicting electricity, hot water, space heating
3. Runs:
   - **Perfect Information**: Optimization with actual data
   - **Forecasted Information**: Optimization with predicted data (with 0% and 20% noise)
4. Compares costs between Perfect and Forecasted information

**Important setting**: In `PREDICTION.py`, the variable `data_directory` at line:
```python
data_directory = os.path.expanduser("~/DB_WORKSPACE")
```
must be changed to point to the correct folder.

---

## Expected Results

### Output of EMS_BB_BFS.py

```
************************ MYOPIC HEURISTIC ************************
binary_fc = ... (COST=...)
binary_pv = ... (COST=...)
...
--- MYOPIC TIME: ... seconds ---

************************ VNS METAHEURISTIC ************************
--IMPROVED BOUND: ITERATION 0, k 0, BEST EVALUATION ...
...

************************ EMS BRANCH & BOUND ************************
>>> WARM START B&B WITH UB = ...

************************ MODEL RESULTS ************************
Fuel Cell Capacity:         ... kW
PV Capacity:                ... kWp
Solar Thermal Area:         ... m²
Heat Pump Capacity:         ... kW
Grid Connection Size:       ... kW
Gas Boiler Capacity:        ... kW
Battery Capacity:           ... kWh
Thermal Tank Height:        ... m
Thermal Tank Volume:        ... Liters

--- COMFORT ---
Hours with penalty (dT>0):  ... / ...
Total penalty cost:         ... CHF/year
Clean energy cost:          ... CHF/year

--- TIMES ---
Myopic Heuristic Time:      ... seconds
VNS Metaheuristic Time:     ... seconds
Branch & Bound Time:        ... seconds

B&B OPTIMAL COST:   ... CHF/year
GAP BETWEEN HEURISTIC AND BB:  ...%
NODES VISITED: ...
```

### Output of PREDICTION.py

```
----- Predict then Optimize: HOUSE3 -----
Train houses: HOUSE3.1, HOUSE3.2
Test house:   HOUSE3
[THESS] Forecast r2 (Electricity): ...   RMSE: ... W
[THESS] Forecast r2 (Hot water):   ...   RMSE: ... W
[THESS] Forecast r2 (Space Heat):  ...   RMSE: ... W

RESULTS [THESS]
Noise Level: 0%
Perfect information:            ... CHF
Forecasted information:         ... CHF
Forecast - Perfect Info:        ... CHF

RESULTS [THESS]
Noise Level: 20%
Perfect information:            ... CHF
Forecasted information:         ... CHF
Forecast - Perfect Info:        ... CHF
```

---

## Troubleshooting

### Problem: "gurobipy module not found"
```bash
pip install gurobipy==12.0.0
```
Ensure that Gurobi Optimizer is installed and the license is activated.

### Problem: "No Gurobi license found"
Follow the instructions at https://www.gurobi.com/academia/academic-program-and-licenses/ for an academic license.

### Problem: FileNotFoundError for CSV files
Ensure that the paths in the scripts point to the correct folder.

### Problem: "numpy<2.0" incompatibility
```bash
pip install "numpy<2.0"
```
The code requires numpy 1.x due to compatibility with older library versions.

---

## LaTeX Compilation

To produce the thesis and presentation PDF:

xelatex  

---

## References

D. Lauinger et al. “A linear programming approach to the optimization of residential energy systems”


---
 
