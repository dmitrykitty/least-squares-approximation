# Fish Weight Prediction: Linear vs. Polynomial Regression

**Author:** Dzmitry, Nikitsin

## Project Overview
This project explores the application of the **Least Squares Method (LSQ)** to predict the weight of fish based on their physical dimensions. The study compares two approaches:
1.  **Linear Regression**: A basic model assuming a linear relationship between dimensions and weight.
2.  **Polynomial Regression (Degree 2)**: An advanced model that introduces interaction terms and squared features to capture non-linear dependencies (geometry/volume).

The goal is to demonstrate the effectiveness of **Feature Engineering** in improving model performance for physical datasets.

## Dataset
The project uses the **Fish Market Dataset** (`data/Fish.csv`).
* **Target Variable ($y$):** `Weight` (g)
* **Features ($X$):**
    * `Length1` (Standard Length)
    * `Length2` (Fork Length)
    * `Length3` (Total Length)
    * `Height`
    * `Width`

## Methodology

### 1. Data Preprocessing
* Loading data using `pandas`.
* Splitting the dataset into **Training (70%)** and **Testing (30%)** sets to evaluate generalization.

### 2. Models Implemented
* **Linear Model:**
    * Uses raw features.
    * Equation: $y = w_0 + w_1x_1 + \dots + w_n x_n$.
    * Solved using the **Moore-Penrose Pseudoinverse matrix**.
* **Polynomial Model:**
    * Uses `sklearn.preprocessing.PolynomialFeatures` (degree=2).
    * Generates new features: $x_i^2$ (squares) and $x_i \cdot x_j$ (interactions).
    * Expands the feature space from 5 to 21 dimensions to capture the volumetric nature of weight.

### 3. Mathematical Foundation
The weights ($\mathbf{w}$) are calculated analytically by minimizing the Sum of Squared Errors (SSE):
$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$
Where $(X^T X)^{-1} X^T$ is the pseudoinverse ($A^+$).

## Results & Analysis

The models were evaluated using **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error).

| Metric   | Linear Model (Test) | Polynomial Model (Test) | Improvement      |
|:---------|:--------------------|:------------------------|:-----------------|
| **RMSE** | 151.35 g            | **75.69 g**             | **~50%**         |
| **MAE**  | 106.24 g            | **48.55 g**             | **~54%**         |
| **MSE**  | 22,907.80           | 5,729.16                | Significant Drop |

### Key Findings:
1.  **Underfitting in Linear Model:** The linear assumption fails to capture the relationship between 1D dimensions (length) and 3D mass (weight), leading to high errors and physically impossible predictions (negative weight for small fish).
2.  **Success of Polynomial Regression:** Introducing polynomial terms allows the model to approximate the volume-weight relationship ($W \propto L^3$). This reduced the error by half.
3.  **Residual Analysis:** The polynomial model's residuals are randomly distributed (white noise), whereas the linear model shows a distinct parabolic pattern indicating structural error.

## Technologies
* **Python 3.x**
* **Libraries:**
    * `numpy` (Matrix operations)
    * `pandas` (Data manipulation)
    * `matplotlib` (Data visualization)
    * `scikit-learn` (Feature engineering & splitting)

## Visualizations
The project includes plots(`data/plots`) for:
* **Actual vs. Predicted** values (visualizing model fit).
* **Residuals** (analyzing error distribution).
* **Feature Analysis** (scatter plots of Weight vs. individual dimensions).