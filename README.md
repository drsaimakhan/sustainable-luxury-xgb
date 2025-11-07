# 🌿 Sustainable Luxury Purchase Prediction (XGBoost + SHAP)

[![View on Kaggle](https://img.shields.io/badge/View%20Notebook-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/code/saimakhan6/sustainable-luxury-purchase-prediction-xg-boost)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-Tool-SHAP-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧭 Overview

Luxury is evolving. Consumers today don’t just buy *status* — they buy *values*.  
This project explores **what drives a consumer’s intention to purchase sustainable luxury products**, using real survey data and explainable AI techniques.

We build a robust **XGBoost classifier** to predict purchase intention and use **SHAP (SHapley Additive exPlanations)** to unpack *why* the model makes its predictions.

---

## 🎯 Business Objective

**Goal:** Identify the psychological and perceptual drivers behind *sustainable luxury purchase intention* to guide:
- Brand strategy & communication
- Market segmentation
- Product and pricing decisions

The project helps luxury marketers answer:
> “Which attitudes, values, and perceptions truly motivate a sustainable luxury purchase — and which ones don’t?”

---

## 🧩 Dataset

| Aspect | Details |
|:--|:--|
| **Source** | Consumer survey (500 respondents) |
| **Target variable** | `purchase_intention` (1 = Likely to buy sustainable luxury) |
| **Features** | Attitudinal and demographic predictors — e.g. sustainability attitude, perceived quality, price sensitivity, social influence, environmental concern |
| **Type** | Tabular survey data (mixed categorical and numeric) |

> Note: The dataset is anonymized and used for educational/research purposes.  
> Raw data is **not included** in this repository for privacy reasons.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Missing value imputation (`SimpleImputer`)
- Categorical encoding (`OneHotEncoder`)
- Feature scaling (`StandardScaler`)
- Combined via `ColumnTransformer` in a clean scikit-learn pipeline

### 2️⃣ Model Building
- **Model:** XGBoost classifier  
- **Hyperparameters:** tuned manually for interpretability (depth, learning rate, regularization)  
- **Evaluation:** Accuracy, F1-score, ROC-AUC

### 3️⃣ Model Explainability
- **SHAP** is used to quantify feature contributions to each prediction
- Global feature importance (summary plot)
- Local explanations (force plot for an individual respondent)

---

## 📈 Results & Insights

| Metric | Score |
|:--|--:|
| **Accuracy** | ~87% |
| **F1-score** | ~0.84 |
| **ROC-AUC** | ~0.90 |

### 🔍 Top Predictors
| Rank | Feature | Interpretation |
|:--|:--|:--|
| 1 | Sustainability Attitude | Strongest driver — eco-conscious consumers show higher purchase intent |
| 2 | Perceived Quality | Quality remains central — sustainability adds *value*, not replaces luxury |
| 3 | Social Influence | Peer approval and social norms shape acceptance of “green luxury” |
| 4 | Price Sensitivity | Moderately negative — price-conscious buyers less likely to purchase |
| 5 | Brand Credibility | Trust and authenticity enhance sustainable perception |

---

## 🧠 Key Takeaways for Managers

- **Consumer mindsets are hybrid:** They seek luxury that aligns with values without sacrificing quality.
- **Authenticity > slogans:** Verified eco-claims, craftsmanship, and transparency outperform generic “green” messaging.
- **Target segment:** High sustainability awareness + quality orientation = ideal audience for premium eco-luxury lines.
- **Communication strategy:**  
  - Emphasize craftsmanship and environmental benefit together.  
  - Use credible messengers (experts, influencers) to reinforce trust.

---

## 📊 Visualizations

| Figure | Description |
|--------|--------------|
| ![SHAP Summary](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_summary.png?raw=true) | **Figure 1. SHAP Summary Plot** — Global view showing the most influential predictors on purchase intention. Higher attitude scores (red) push predictions toward “likely to buy,” while lower ones (blue) reduce likelihood. |
| ![SHAP Dependence](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_dependence_att1.png?raw=true) | **Figure 2. SHAP Dependence Plot for att1** — Relationship between `att1` values and their SHAP impact, colored by `att2`. Highlights nonlinear effects. |
| ![Local Bar Plot](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_bar_local.png?raw=true) | **Figure 3. Local SHAP Bar Plot** — Feature-level explanation for one respondent, showing positive (red) and negative (blue) contributions. |
| ![Force Plot](https://github.com/drsaimakhan/sustainable-luxury-xgb/blob/main/assets/shap_force_local.png?raw=true) | **Figure 4. SHAP Force Plot** — Detailed local explanation illustrating how each variable moves the prediction away from the base value. |

---
## 🛠️ How to Reproduce

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/sustainable-luxury-xgb.git
cd sustainable-luxury-xgb
pip install -r requirements.txt
jupyter notebook notebooks/Sustainable_Luxury_Purchase_Prediction_XGBoost.ipynb
