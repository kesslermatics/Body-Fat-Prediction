# Body Fat Prediction using Linear Regression and Neural Networks

## 📌 Project Overview

This project explores the prediction of body fat percentage based on anthropometric measurements using both **Linear Regression** and a **Neural Network (NN)** model. The primary objective is to evaluate the predictive capability of interpretable statistical models versus flexible machine learning approaches when limited to a small set of body measurements. This task is especially relevant for health monitoring and body composition estimation in scenarios where sophisticated body fat measurement methods (e.g., underwater weighing or DEXA) are not feasible.


## 🎯 Objective

The goal is to estimate **body fat percentage** from features such as weight, height, circumferences of various body parts (e.g., abdomen, thigh, wrist), and derived metrics such as **BMI** and **Waist-Hip Ratio**, without relying on direct body composition scans or density-based methods.



## 📚 Dataset

- **Source:** [Kaggle Dataset: BMI and BodyFat Data](https://www.kaggle.com/datasets/yasserh/bmidataset)
- **Features include:**
  - Demographic data (age)
  - Anthropometric data (weight, height, wrist, abdomen, etc.)
  - Density (from underwater weighing, excluded to prevent leakage)
  - Derived features (BMI, Waist-Hip Ratio)

The target variable (`BodyFat`) is calculated from **Siri’s equation** based on density



## 🧪 Methodology

### 1. **Linear Regression**
- **Rationale:** Simple, interpretable, and serves as a robust baseline.
- **Features used:** After exploratory analysis, a reduced set of features was chosen:  
  `Wrist`, `Neck`, `Forearm`, `Thigh`, `Hip`, `Biceps`, `Ankle`, and two engineered features:  
  - **BMI** (Body Mass Index)
  - **Waist-Hip Ratio**
- **Results:**
  - R² Score: `0.6546`
  - MSE: `16.07`

This indicates that approximately **65% of the variance** in body fat percentage can be explained by the selected features. While the model performs reasonably well, a substantial portion of variance remains unexplained — potentially due to missing physiological factors not captured in the dataset.

### 2. **Neural Network (PyTorch)**
- **Rationale:** Allows capturing complex, non-linear relationships between inputs and target.
- **Architecture:**
  - 2 hidden layers: 64 and 32 neurons
  - Activation: ReLU
  - Optimizer: Adam
  - Loss: Mean Squared Error
  - Epochs: 100
- **Feature normalization (StandardScaler)** was applied to improve convergence.

- **Key Metrics:**
  - Final Train MSE: `14.50`
  - Final Validation MSE: `16.78`
  - Final Validation MAE: `3.33`

Despite the higher modeling capacity of the neural network, **the performance gain over linear regression was marginal**. Validation loss plateaued early, and the learning curve showed no signs of overfitting — which suggests the **data itself imposes a limit** on prediction accuracy.


## 🔍 Observations & Interpretation

- Many input features (e.g., height, knee circumference) showed little to no correlation with `BodyFat`, while `Abdomen` and `Wrist` emerged as the strongest individual predictors.
- The **NN model slightly improved training loss**, but **validation loss remained close** to the linear regression baseline, implying that **model complexity is not the bottleneck**.
- Given the nature of the data and the biological variability in fat distribution, we conclude that a prediction error around **3–4% BodyFat (MAE)** is **an acceptable lower bound** in this context.


## 🧠 Conclusion

This project demonstrates that **carefully selected and engineered features** can yield strong predictive performance even with simple models. While neural networks offer flexibility, they **do not always outperform** interpretable models when data is limited or noisy. The results also highlight the **inherent limits of predictability** for biological traits when constrained by observational (non-invasive) data.
