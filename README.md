# Robust_ML_Retail_project
# Robust‑ML Retail‑Sales Prediction Experiment
Portfolio mini‑project: Test whether Random‑Forest delivers more‑robust price‑prediction than a simple Decision‑Tree when real‑world‑style data defects are introduced.

## Dataset
UCI Online Retail dataset (UK‑retail transaction records). Pre‑processed: kept only positive‑value sales rows; added TotalPrice calculated column.

## Workflow
1. Data cleaning: filtered return‑transaction rows; converted date and numeric fields
2. Created modified‑quality variants: 5 % missing‑Quantity entries, 3 % Unit‑Price outliers
3. Trained Decision‑Tree (baseline) & Random‑Forest (robust‑candidate) regression models
4. Compared RMSE prediction error across three dataset‑quality levels
5. Visualised error‑comparison results

## Key findings
- Dataset first prepared in Excel before machine‑learning‑based robustness testing in Python.
‑ Both models handled moderate missing‑Quantity values well (low RMSE)
‑ Prediction error rose sharply when Unit‑Price outliers existed
‑ In this experiment, baseline Decision‑Tree showed slightly‑lower RMSE than Random‑Forest ‑ indicating need for further forest‑model tuning

## Project files
‑ robust_analysis.py ‑ full Python‑analysis script
‑ online_retail_clean.csv ‑ cleaned‑transaction dataset
‑ rmse_comparison_chart.png ‑ RMSE‑comparison bar‑chart
‑ project_summary.txt ‑ brief‑experiment note

## Note:
The cleaned CSV file is not included in‑repo due to file‑size limits. The analysis script contains all pre‑processing steps; the original public dataset is the UCI Online Retail dataset from the UCI Machine‑Learning Repository.

## Tools used:

Python ‑ Pandas, NumPy, Matplotlib, Scikit‑learn (Machine‑Learning modelling); Microsoft Excel ‑ initial dataset preparation, calculated‑column creation
---
## Prepared by Shumaila
