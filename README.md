# Probing the Statistical Frontiers of Early-Risk Prediction

This repository contains the complete Python analysis code for the study:

**“Probing the Statistical Frontiers of Early-Risk Prediction: A Methodological Framework for Deconstructing Biomarker Effects in a Gold-Standard Healthy Cohort.”**

This study introduces a methodological framework for biomarker discovery in low-risk populations, demonstrating its application by identifying core sleep biomarkers for predicting early cardiometabolic risk.

---

## 📄 Abstract

Predicting incipient cardiometabolic disease in truly healthy individuals is a frontier challenge. Conventional models often fail in these low-risk settings. This study presents a methodological blueprint for rigorously investigating weak signals in preventive medicine. By strategically combining:

- An **extreme phenotype design** (the *Gold-Standard Healthy Cohort*),
- **Outcome deconstruction**, and
- **Proactive power analysis**,

we transform statistical limitations into scientific insights.

Using this framework, we demonstrate that **sleep-disordered breathing (SDB)** and **nocturnal hypoxemia** are valid, but diffuse, early risk factors, and define the statistical conditions required to robustly detect their pathway-specific effects.

---

## 📊 Data Source

> **Important Note**  
> This repository does **not** include raw data.  
> Due to data use agreements and participant privacy protections, the original datasets used in this study cannot be redistributed.

The data used in this study are from the **Sleep Heart Health Study (SHHS)**.  
You may apply for data access through the official **National Sleep Research Resource (NSRR)**:

🔗 [https://sleepdata.org](https://sleepdata.org)

---

## 💻 Code Description

**Main script:**

- `step10_complete_analysis_subend.py`:  
  A self-contained Python script covering:
  - Data loading  
  - Cohort definition  
  - Model training and evaluation  
  - Statistical analysis  
  - Generation of all final figures and tables

**Core dependencies:**

```
pandas  
numpy  
scikit-learn  
matplotlib  
seaborn  
shap  
statsmodels  
xgboost
```

---

## ▶️ How to Run

### 1. Environment Setup

Install Python (**3.9+** recommended), and install the required packages:

```
pip install pandas numpy scikit-learn matplotlib seaborn shap statsmodels xgboost
```

### 2. Data Preparation

Download the SHHS dataset from NSRR and place the following files in the same folder as the script:

- `shhs1-dataset-0.21.0.csv`  
- `shhs2-dataset-0.21.0.csv`

### 3. Execute the Script

```
python step10_complete_analysis_subend.py
```

---

## ✅ Expected Output

After successful execution, the script will print:

- Final sample size (**n = 447**)
- Cross-validated AUCs and statistical *p*-values
- IDI and adjusted ORs with 95% CIs
- Coefficients from the final LASSO model
- Baseline characteristics (Table 1)

All figures used in the paper will be saved as `.png` files in a folder named `output_figures`.

---

## 📚 Citation

If you use this repository in your research, please cite our paper (to be updated after publication):

> _Citation details coming soon._

---

## 📝 License

This project is licensed under the **MIT License**.

