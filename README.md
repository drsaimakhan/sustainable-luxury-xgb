# 🌿 Sustainable Luxury Purchase Prediction (XGBoost + SHAP)

[![View on Kaggle](https://img.shields.io/badge/View%20Notebook-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/code/saimakhan6/sustainable-luxury-purchase-prediction-xg-boost)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Model: XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Explainability: SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧭 Overview

Luxury is evolving — consumers now seek **values alongside status**.  
This project analyzes *what drives the intention to purchase sustainable luxury products* using machine learning and explainable AI.

An **XGBoost classifier** predicts purchase intention from survey data, while **SHAP (SHapley Additive exPlanations)** reveals which attitudes, values, and personality traits shape those predictions.

---

## 🎯 Business Objective

**Goal:** Understand the psychological and perceptual factors that influence a consumer’s likelihood of buying sustainable luxury.  
Insights help brands optimize:
- **Targeting:** Identify high-intent consumers  
- **Messaging:** Emphasize sustainability authenticity and craftsmanship  
- **Product & pricing:** Balance eco-value with perceived luxury  

---

## 🧩 Dataset

| Aspect | Description |
|:--|:--|
| **Source** | Consumer survey (500 respondents) |
| **Target variable** | `purchase_intention` (1 = Likely to buy sustainable luxury) |
| **Predictors** | Attitudes (`att1–att5`), Big-Five personality scores, motivational drivers, demographics |
| **Type** | Tabular survey data (categorical + numeric) |
| **Privacy** | Anonymized data used solely for research/educational purposes |

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Missing values handled with `SimpleImputer`  
- Categorical encoding via `OneHotEncoder`  
- Scaling with `StandardScaler`  
- Combined through a `ColumnTransformer` in a single pipeline  

### 2️⃣ Model Building
- Algorithm: **XGBoost Classifier**  
- Parameters tuned for interpretability (depth = 4, learning rate = 0.06, λ = 1.0)  
- Evaluation metrics: Accuracy, F1-score, ROC-AUC  

### 3️⃣ Explainability
- **SHAP** used to compute global and local feature contributions  
- Visualization suite includes summary, dependence, bar, and force plots  

---

## 📈 Results

| Metric | Score |
|:--|--:|
| **Accuracy** | ~0.87 |
| **F1-Score** | ~0.84 |
| **ROC-AUC** | ~0.90 |

**Top Predictors**
1. Attitudinal variables (`att1–att5`)
2. Perceived quality
3. Social influence
4. Motivational drivers
5. Personality openness & agreeableness

---

## 🧠 Model Explainability & Visualizations

Understanding *why* the model predicts a consumer’s intention is as critical as its accuracy.  
We used **SHAP** to interpret both *global importance* and *individual-level reasoning* behind predictions.

These plots demonstrate how each variable contributes to or detracts from purchase intention.

| Figure | Description |
|--------|--------------|
| ![SHAP Summary](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_summary.png?raw=true) | **Figure 1 – SHAP Summary Plot**: Global importance of predictors. Higher attitude scores (red) push predictions toward “likely to buy,” while lower ones (blue) reduce intent. |
| ![SHAP Dependence](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_dependence_att1.png?raw=true) | **Figure 2 – SHAP Dependence Plot (att1 vs att2)**: Shows positive nonlinear relationship between attitude and predicted intent. |
| ![Local Bar Plot](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_bar_local.png?raw=true) | **Figure 3 – Local SHAP Bar Plot**: Individual explanation highlighting which features increased (red) or decreased (blue) predicted intent. |
| ![Force Plot](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_force_local.png?raw=true) | **Figure 4 – SHAP Force Plot**: Visualization of a single prediction; shows how features push the model output above or below the base value. |

### 🔍 Managerial Interpretation

- **Attitudinal factors** are the most powerful levers of sustainable purchase intent.  
- **Motivational and personality drivers** add psychological nuance.  
- Local SHAP explanations enable segment-level or personalized strategies.  

Together, these explainability insights translate raw ML output into actionable marketing intelligence.

---

## 💼 Strategic Takeaways

- **Authenticity > slogans:** verified eco-claims, craftsmanship, and transparency outperform generic “green” messaging.  
- **High sustainability + quality orientation = prime target segment.**  
- **Channel strategy:** use credible voices (experts, creators) to reinforce trust and social validation.  

---

## 🧮 Repository Structure

