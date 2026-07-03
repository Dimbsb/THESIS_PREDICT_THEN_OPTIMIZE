# Predict-then-Optimize Προσέγγιση για Ενεργειακά Συστήματα

## 📋 Περιγραφή

Διπλωματική εργασία με τίτλο **"Predict then Optimize Προσέγγιση για Ενεργειακά Συστήματα"** του Μπίσμπα Δημητρίου, στο Τμήμα Ηλεκτρολόγων Μηχανικών & Μηχανικών Υπολογιστών του Πανεπιστημίου Δυτικής Μακεδονίας.

Στόχος είναι η βέλτιστη διαστασιολόγηση και λειτουργία ενεργειακών συστημάτων κατοικιών (Fuel Cell, Φ/Β, Ηλιακά Θερμικά, Αντλία Θερμότητας, Boiler, Μπαταρία, Δεξαμενή Θερμότητας) ελαχιστοποιώντας το ετήσιο κόστος, μέσω:
- **Πρόβλεψης** ζήτησης (Random Forest) για ηλεκτρισμό, ζεστό νερό και θέρμανση χώρου
- **Βελτιστοποίησης** με MILP (Mixed Integer Linear Programming) χρησιμοποιώντας:
  - Εμπορικό επιλυτή **Gurobi**
  - Ιδιόχειρο αλγόριθμο **Branch & Bound με Best-First Search**
  - Ευρετικές/Μετα-ευρετικές μεθόδους (Myopic, VNS)

---

## 📂 Δομή Αρχείων

```
.
├── README.md                           # Το παρόν αρχείο
├── thesis.pdf                          # Το πλήρες κείμενο της διπλωματικής (165 σελίδες)
│
├── latex_sources/                      # Οι πηγαίοι κώδικες LaTeX
│   ├── thesis.tex                      # Κύριο αρχείο διπλωματικής
│   ├── presentation.tex                # Παρουσίαση (Beamer)
│   ├── appendix_all.tex                # Παραρτήματα
│   ├── refs.bib                        # Βιβλιογραφία (BibTeX)
│   └── images/                         # Εικόνες και διαγράμματα
│
├── code/                               # Κώδικας Python
│   ├── EMS_MODEL.py                    # Μοντέλο MILP (Gurobi) - ορισμός μεταβλητών, περιορισμών, αντικειμενικής
│   ├── EMS_GUROBI.py                   # Εναλλακτική υλοποίηση με Pyomo + Gurobi
│   ├── EMS_BB_BFS.py                   # Branch & Bound + Myopic + VNS + solve_instance()
│   ├── PREDICTION.py                   # Πρόβλεψη φορτίων με Random Forest + Predict-then-Optimize pipeline
│   └── DATA_GENERATOR.ipynb            # Jupyter notebook για την παραγωγή δεδομένων (OCHRE simulator)
│
├── data/                               # Δεδομένα κατοικιών (OCHRE_DATA)
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
└── weather/                            # Μετεωρολογικά αρχεία EPW
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

**ΣΗΜΑΝΤΙΚΟ**: Απαιτείται άδεια χρήσης **Gurobi Optimizer**. Μπορείτε να αποκτήσετε:
- **Ακαδημαϊκή άδεια**: Δωρεάν μέσω του [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)
- **Δωρεάν δοκιμή**: 30 ημέρες μέσω του [Gurobi Trial](https://www.gurobi.com/downloads/)

Μετά την εγκατάσταση, βεβαιωθείτε ότι η άδεια είναι ενεργοποιημένη:
```bash
gurobi_cl --version
```

### 3. Εγκατάσταση Βιβλιοθηκών

```bash
pip install gurobipy==12.0.0
pip install "numpy<2.0"
pip install pandas scikit-learn pyomo matplotlib seaborn
```

### 4. LaTeX (προαιρετικό, για μεταγλώττιση του PDF)

Απαιτείται **XeLaTeX** (λόγω ελληνικών χαρακτήρων):
```bash
sudo apt-get install texlive-xetex texlive-lang-greek texlive-bibtex-extra
```

---

## 🚀 Οδηγίες Εκτέλεσης

### Βήμα 1: Ρύθμιση δεδομένων

Τα δεδομένα βρίσκονται ήδη έτοιμα στον φάκελο `data/`. Αν θέλετε να τα παράγετε από την αρχή:

1. Ανοίξτε το `code/DATA_GENERATOR.ipynb` στο Jupyter Notebook ή στο VS Code
2. Εκτελέστε τα cells με τη σειρά:
   - **Cell 1**: Εγκατάσταση εξαρτήσεων (OCHRE, numpy, gurobipy)
   - **Cell 2**: Αντιγραφή του προτύπου XML του OCHRE
   - **Cell 3**: Δημιουργία XML αρχείων για κάθε κατοικία/πόλη
   - **Cell 4**: Προσομοίωση για Θεσσαλονίκη (παράγει τα CSV)
   - **Cell 5**: Προσομοίωση για Αθήνα (παράγει τα CSV με θόρυβο)

**Προσοχή**: Το DATA_GENERATOR.ipynb έχει σχεδιαστεί για Google Colab (χρησιμοποιεί διαδρομές `/content/`). Για τοπική εκτέλεση θα χρειαστούν τροποποιήσεις στις διαδρομές.

### Βήμα 2: Επίλυση με Gurobi (Perfect Information)

Για να τρέξετε τον εμπορικό επιλυτή Gurobi:

```bash
cd code/
python EMS_GUROBI.py --house HOUSE3 --location THESS --data-dir ../data --hours 8760
```

**Παράμετροι**:
- `--house`: Κατοικία (HOUSE1, HOUSE2, HOUSE3)
- `--location`: Πόλη (ATH, THESS)
- `--data-dir`: Διαδρομή προς τον φάκελο δεδομένων
- `--hours`: Χρονικός ορίζοντας σε ώρες (8760 για ένα έτος)

### Βήμα 3: Branch & Bound (Custom Solver)

Για τον ιδιόχειρο αλγόριθμο Branch & Bound με Myopic και VNS:

```bash
cd code/
python EMS_BB_BFS.py
```

**Σημείωση**: Το αρχείο `EMS_BB_BFS.py` στο κυρίως block (`if __name__ == "__main__"`) είναι ρυθμισμένο για HOUSE3/THESS. Τροποποιήστε τη γραμμή:

```python
data = load_data(location="THESS", T=8760, data_directory=os.path.expanduser("~/DB_WORKSPACE/HOUSE3"))
```

Αλλάξτε το `data_directory` ώστε να δείχνει στον σωστό φάκελο, π.χ.:
```python
data = load_data(location="THESS", T=8760, data_directory="../data/HOUSE3")
```

### Βήμα 4: Predict-then-Optimize (Πρόβλεψη + Βελτιστοποίηση)

Για την πλήρη ροή πρόβλεψης-βελτιστοποίησης:

```bash
cd code/
python PREDICTION.py
```

**Τι κάνει**:
1. Φορτώνει δεδομένα από τα train houses (π.χ. HOUSE3.1, HOUSE3.2) και το test house (HOUSE3)
2. Εκπαιδεύει Random Forest μοντέλα για πρόβλεψη ηλεκτρισμού, DHW, θέρμανσης
3. Τρέχει:
   - **Perfect Information**: Βελτιστοποίηση με τα πραγματικά δεδομένα
   - **Forecasted Information**: Βελτιστοποίηση με τα προβλεπόμενα δεδομένα (με 0% και 20% θόρυβο)
4. Συγκρίνει το κόστος μεταξύ Perfect και Forecasted πληροφορίας

**Σημαντική ρύθμιση**: Στο `PREDICTION.py`, η μεταβλητή `data_directory` στη γραμμή:
```python
data_directory = os.path.expanduser("~/DB_WORKSPACE")
```
πρέπει να αλλάξει ώστε να δείχνει στον φάκελο `data/` του project:
```python
data_directory = "../data"
```

---

## 📊 Αναμενόμενα Αποτελέσματα

### Έξοδος EMS_BB_BFS.py

```
************************ MYOPIC HEURISTIC ************************
binary_fc = 1 (COST=1,234.56)
binary_pv = 1 (COST=1,100.00)
...
--- MYOPIC TIME: 12.34 seconds ---

************************ VNS METAHEURISTIC ************************
--IMPROVED BOUND: ITERATION 0, k 0, BEST EVALUATION 1050.12345
...

************************ EMS BRANCH & BOUND ************************
>>> WARM START B&B WITH UB = 1050.12

************************ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ************************
Fuel Cell Capacity:         1.50 kW
PV Capacity:                3.20 kWp
Solar Thermal Area:         2.10 m²
Heat Pump Capacity:         2.80 kW
Grid Connection Size:       4.50 kW
Gas Boiler Capacity:        0.00 kW
Battery Capacity:           5.20 kWh
Thermal Tank Height:        1.50 m
Thermal Tank Volume:        754 Liters

--- COMFORT ---
Hours with penalty (dT>0):  124 / 8760
Total penalty cost:         15.30 CHF/year
Clean energy cost:          1034.82 CHF/year

--- TIMES ---
Myopic Heuristic Time:      12.34 seconds
VNS Metaheuristic Time:     25.67 seconds
Branch & Bound Time:        45.12 seconds

B&B OPTIMAL COST:   1045.12 CHF/year
GAP BETWEEN HEURISTIC AND BB:  0.4785%
NODES VISITED: 47
```

### Έξοδος PREDICTION.py

```
----- Predict then Optimize: HOUSE3 -----
Train houses: HOUSE3.1, HOUSE3.2
Test house:   HOUSE3
[THESS] Forecast r2 (Electricity): 0.8234   RMSE: 125.67 W
[THESS] Forecast r2 (Hot water):   0.7890   RMSE: 89.34 W
[THESS] Forecast r2 (Space Heat):  0.9012   RMSE: 234.56 W

ΑΠΟΤΕΛΕΣΜΑΤΑ [THESS]
Noise Level: 0%
Perfect information:            1,045.12 CHF
Forecasted information:         1,089.34 CHF
Forecast - Perfect Info:        44.22 CHF

ΑΠΟΤΕΛΕΣΜΑΤΑ [THESS]
Noise Level: 20%
Perfect information:            1,045.12 CHF
Forecasted information:         1,145.67 CHF
Forecast - Perfect Info:        100.55 CHF
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
Βεβαιωθείτε ότι οι διαδρομές στα scripts δείχνουν σωστά στον φάκελο `data/`. Ελέγξτε:
- `EMS_BB_BFS.py`: `data_directory` στο `load_data()`
- `PREDICTION.py`: `data_directory` στη γραμμή 105
- `EMS_GUROBI.py`: `--data-dir` argument

### Πρόβλημα: "numpy<2.0" incompatibility
```bash
pip install "numpy<2.0"
```
Ο κώδικας απαιτεί numpy 1.x λόγω συμβατότητας με παλαιότερες εκδόσεις βιβλιοθηκών.

---

## 📝 Μεταγλώττιση LaTeX

Για την παραγωγή του PDF της διπλωματικής:

```bash
cd latex_sources/
xelatex thesis.tex
bibtex thesis
xelatex thesis.tex
xelatex thesis.tex
```

Για την παρουσίαση:
```bash
cd latex_sources/
xelatex presentation.tex
xelatex presentation.tex
```

---

## 📚 Αναφορές

- Ashouri, A. (2014). *Simultaneous Design and Control of Energy Systems*. PhD Thesis, ETH Zürich.
- Petrousov, I. & Ploskas, N. — Πρότυπο LaTeX διπλωματικής εργασίας, Πανεπιστήμιο Δυτικής Μακεδονίας.

---

## 👨‍💻 Στοιχεία Επικοινωνίας

**Φοιτητής**: Μπίσμπας Δημήτριος (ΑΜ: 2037)  
**Επιβλέπων**: Πλόσκας Νικόλαος  
**Τμήμα**: Ηλεκτρολόγων Μηχανικών & Μηχανικών Υπολογιστών  
**Πανεπιστήμιο**: Πανεπιστήμιο Δυτικής Μακεδονίας  
**Εργαστήριο**: Εργαστήριο Ευφυών Συστημάτων και Βελτιστοποίησης


