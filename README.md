# Predict-then-Optimize Προσέγγιση για Ενεργειακά Συστήματα

## 📋 Περιγραφή

Διπλωματική εργασία με τίτλο **"Predict then Optimize Προσέγγιση για Ενεργειακά Συστήματα"** του Μπίσμπα Δημητρίου, στο Τμήμα Ηλεκτρολόγων Μηχανικών & Μηχανικών Υπολογιστών του Πανεπιστημίου Δυτικής Μακεδονίας.
 
Στόχος είναι η βέλτιστη διαστασιολόγηση και λειτουργία ενεργειακών συστημάτων κατοικιών (κυψέλη καυσίμου, φωτοβολταϊκά, θερμοσίφωνας, αντλία θερμότητας, λέβητας φυσικού αερίου, μπαταρία, θερμική δεξαμενή) ελαχιστοποιώντας το ετήσιο κόστος, μέσω:
- **Πρόβλεψης** ζήτησης (Random Forest) για ηλεκτρισμό, ζεστό νερό και θέρμανση χώρου
- **Βελτιστοποίησης** με MILP (Mixed Integer Linear Programming) χρησιμοποιώντας:
  - Εμπορικό επιλυτή **Gurobi**
  - Custom **Branch & Bound με Best-First Search** με ευρετικές και μεθευρετικές μεθόδους (Myopic, VNS)

---

## 📂 Δομή Αρχείων

```
.
├── README.md                           # Το παρόν αρχείο
├── THESIS_PREDICT_THEN_OPTIMIZE.pdf    # Το κείμενο της διπλωματικής (165 σελίδες)
├── THESIS_PPTX.pdf                     # Το power point της διπλωματικής  
│
├── LATEX_SOURCES/                      # Οι πηγαίοι κώδικες LaTeX
│   ├── thesis.tex                      # Κύριο αρχείο διπλωματικής
│   ├── presentation.tex                # Παρουσίαση (Beamer)
│   ├── appendix_all.tex                # Παραρτήματα
│   ├── refs.bib                        # Βιβλιογραφία (BibTeX)
│   └── images/                         # Εικόνες και διαγράμματα
│       Λοιπά αρχεία:
│       ├── thesis.loa 
│       ├── thesis.lof 
│       ├── thesis.lol
│       └── thesis.lot
│         
│
├── MODEL_SOLVER_CODES/                 # Κώδικας Python
│   ├── EMS_MODEL.py                    # Μοντέλο MILP - ορισμός μεταβλητών, περιορισμών, αντικειμενικής
│   ├── EMS_GUROBI.py                   # Υλοποίηση με Pyomo + Gurobi
│   ├── EMS_BB_BFS.py                   # Branch & Bound + Myopic + VNS + solve_instance()
│   ├── PREDICTION.py                   # Πρόβλεψη ζήτησης με Random Forest + Predict-then-Optimize pipeline
│   └── DATA_GENERATOR.ipynb            # Jupyter notebook για την παραγωγή δεδομένων (OCHRE simulator)
│
├── OCHRE_DATA/                         # Δεδομένα κατοικιών (OCHRE_DATA)
│   ├── HOUSE1/                         # Δεδομένα δοκιμής (test) για Κατοικία 1
│   ├── HOUSE1.1/                       # Δεδομένα εκπαίδευσης (train) για Κατοικία 1
│   ├── HOUSE1.2/                       # Δεδομένα εκπαίδευσης (train) για Κατοικία 1
│   ├── HOUSE2/                         # Δεδομένα δοκιμής (test) για Κατοικία 2
│   ├── HOUSE2.1/
│   ├── HOUSE2.2/
│   ├── HOUSE3/                         # Δεδομένα δοκιμής (test) για Κατοικία 3
│   ├── HOUSE3.1/
│   └── HOUSE3.2/
│       Σε κάθε φάκελο:
│       ├── ELECTRICITY_{ATH|THESS}.csv   # Ωριαία ζήτηση ηλεκτρισμού (W)
│       ├── DHW_{ATH|THESS}.csv           # Ωριαία ζήτηση ζεστού νερού (W)
│       ├── SPACE_HEAT_{ATH|THESS}.csv    # Ωριαία ζήτηση θέρμανσης χώρου (W)
│       ├── SOLAR_DATA_{ATH|THESS}.csv    # Ωριαία ηλιακή ακτινοβολία GHI (W/m²)
│       └── TEMPERATURES_{ATH|THESS}.csv  # Θερμοκρασίες περιβάλλοντος και συλλέκτη (°C)
│
└── WEATHER_DATA_ENERGY+/                # Μετεωρολογικά αρχεία EPW
    ├── GRC_Athens.167160_IWEC.epw       # Καιρικά δεδομένα Αθήνας
    └── GRC_Thessaloniki.166220_IWEC.epw # Καιρικά δεδομένα Θεσσαλονίκης
```

---

## ⚙️ Απαιτήσεις Λογισμικού

### 1. Python και Βιβλιοθήκες

Απαιτείται **Python 3.9+**. Οι κύριες βιβλιοθήκες είναι:

| Βιβλιοθήκη | Χρήση |
|------------|-------|
| `gurobipy==12.0.0` | Εμπορικός επιλυτής MILP |
| `numpy<2.0` | Αριθμητικές πράξεις |
| `pandas` | Διαχείριση δεδομένων |
| `scikit-learn` | Random Forest Regressor για προβλέψεις |
| `pyomo` | Modeling γλώσσα (εναλλακτική του gurobipy) |
| `matplotlib` / `seaborn` | Οπτικοποίηση αποτελεσμάτων |

### 2. Gurobi Optimizer

- **Ακαδημαϊκή άδεια** 

### 3. Εγκατάσταση Βιβλιοθηκών

```bash
pip install gurobipy==12.0.0
pip install "numpy<2.0"
pip install pandas scikit-learn pyomo matplotlib seaborn
```

### 4. LaTeX (μεταγλώττιση του PDF)

Απαιτείται **XeLaTeX**

---

## 🚀 Οδηγίες Εκτέλεσης

### Βήμα 1: Ρύθμιση δεδομένων

Τα δεδομένα βρίσκονται ήδη έτοιμα στον φάκελο `OCHRE_DATA/`. Αν θέλετε να τα παράγετε από την αρχή:

1. Ανοίξτε το `code/DATA_GENERATOR.ipynb` στο Google Colab
2. Εκτελέστε τα cells με τη σειρά:
   - **Cell 1**: Εγκατάσταση (OCHRE, numpy, gurobipy)
   - **Cell 2**: Αντιγραφή του προτύπου XML του OCHRE
   - **Cell 3**: Δημιουργία XML αρχείων για κάθε κατοικία/πόλη
   - **Cell 4**: Παράγει τα CSV
   - **Cell 5**: Παράγει τα CSV για τις πανομοιότυπες κατοικίες

Το DATA_GENERATOR.ipynb έχει σχεδιαστεί για Google Colab (χρησιμοποιεί διαδρομές `/content/`). Για τοπική εκτέλεση θα χρειαστούν τροποποιήσεις στις διαδρομές.

### Βήμα 2: Επίλυση με Gurobi (Perfect Information)

Για να τρέξετε τον εμπορικό επιλυτή Gurobi:

```bash
python EMS_GUROBI.py --house HOUSE3 --location THESS --data-dir ../data --hours 8760
```

**Παράμετροι**:
- `--house`: Κατοικία (HOUSE1, HOUSE2, HOUSE3)
- `--location`: Πόλη (ATH, THESS)
- `--data-dir`: Διαδρομή προς τον φάκελο δεδομένων
- `--hours`: Χρονικός ορίζοντας σε ώρες (8760 για ένα έτος)

### Βήμα 3: Branch & Bound (Custom Solver)

Για τον custom αλγόριθμο Branch & Bound με Myopic και VNS:

```bash
python EMS_BB_BFS.py
```

**Σημείωση**: Το αρχείο `EMS_BB_BFS.py` στο κυρίως block (`if __name__ == "__main__"`) είναι ρυθμισμένο για HOUSE3/THESS. Τροποποιήστε τη γραμμή:

```python
data = load_data(location="THESS", T=8760, data_directory=os.path.expanduser("~/DB_WORKSPACE/HOUSE3"))
```

Αλλάξτε το `data_directory` ώστε να δείχνει στον σωστό φάκελο.

### Βήμα 4: Predict-then-Optimize (Πρόβλεψη + Βελτιστοποίηση)

Για την πλήρη ροή πρόβλεψης-βελτιστοποίησης:

```bash
python PREDICTION.py
```

**Ροή**:
1. Φορτώνει δεδομένα από τα train houses (π.χ. HOUSE3.1, HOUSE3.2) και το test house (HOUSE3)
2. Εκπαιδεύει Random Forest μοντέλα για πρόβλεψη ηλεκτρισμού, ζεστού νερού, θέρμανσης
3. Τρέχει:
   - **Perfect Information**: Βελτιστοποίηση με τα πραγματικά δεδομένα
   - **Forecasted Information**: Βελτιστοποίηση με τα προβλεπόμενα δεδομένα (με 0% και 20% θόρυβο)
4. Συγκρίνει το κόστος μεταξύ Perfect και Forecasted πληροφορίας

**Σημαντική ρύθμιση**: Στο `PREDICTION.py`, η μεταβλητή `data_directory` στη γραμμή:
```python
data_directory = os.path.expanduser("~/DB_WORKSPACE")
```
πρέπει να αλλάξει ώστε να δείχνει στον σωστό φάκελο.

---

## 📊 Αναμενόμενα Αποτελέσματα

### Έξοδος EMS_BB_BFS.py

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

************************ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ************************
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

### Έξοδος PREDICTION.py

```
----- Predict then Optimize: HOUSE3 -----
Train houses: HOUSE3.1, HOUSE3.2
Test house:   HOUSE3
[THESS] Forecast r2 (Electricity): ...   RMSE: ... W
[THESS] Forecast r2 (Hot water):   ...   RMSE: ... W
[THESS] Forecast r2 (Space Heat):  ...   RMSE: ... W

ΑΠΟΤΕΛΕΣΜΑΤΑ [THESS]
Noise Level: 0%
Perfect information:            ... CHF
Forecasted information:         ... CHF
Forecast - Perfect Info:        ... CHF

ΑΠΟΤΕΛΕΣΜΑΤΑ [THESS]
Noise Level: 20%
Perfect information:            ... CHF
Forecasted information:         ... CHF
Forecast - Perfect Info:        ... CHF
```

---

## 🔧 Αντιμετώπιση Προβλημάτων

### Πρόβλημα: "gurobipy module not found"
```bash
pip install gurobipy==12.0.0
```
Βεβαιωθείτε ότι το Gurobi Optimizer είναι εγκατεστημένο και η άδεια ενεργοποιημένη.

### Πρόβλημα: "No Gurobi license found"
Ακολουθήστε τις οδηγίες στο https://www.gurobi.com/academia/academic-program-and-licenses/ για ακαδημαϊκή άδεια.

### Πρόβλημα: FileNotFoundError για τα CSV αρχεία
Βεβαιωθείτε ότι οι διαδρομές στα scripts δείχνουν στον σωστό φάκελο.

### Πρόβλημα: "numpy<2.0" incompatibility
```bash
pip install "numpy<2.0"
```
Ο κώδικας απαιτεί numpy 1.x λόγω συμβατότητας με παλαιότερες εκδόσεις βιβλιοθηκών.

---

## 📝 Μεταγλώττιση LaTeX

Για την παραγωγή του PDF της διπλωματικής και της παρουσίασης:

xelatex  

---

## 📚 Αναφορές

D. Lauinger et al. “A linear programming approach to the optimization of residential energy systems”


---
 
