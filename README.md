# ft_linear_regression

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![42 Curriculum](https://img.shields.io/badge/42_Curriculum-Post--Common_Core-000000?style=for-the-badge)](https://42.fr/)
[![Code Style](https://img.shields.io/badge/Code_Style-flake8%20%7C%20PEP_8-brightgreen?style=for-the-badge)](https://flake8.pycqa.org/)
[![ML Dependencies](https://img.shields.io/badge/ML_Frameworks-Zero_Dependencies-red?style=for-the-badge)]()
[![Optimization](https://img.shields.io/badge/Optimizer-Batch_Gradient_Descent-blue?style=for-the-badge)]()

> A production-grade, framework-free implementation of univariate linear regression featuring vectorized batch gradient descent, dynamic min-max feature scaling with analytical parameter denormalization, decoupled inference, and statistical goodness-of-fit ($R^2$) validation.

---

## 📌 Academic / Project Context

`ft_linear_regression` is the foundational machine learning project in the **42 School Post-Common Core curriculum** (Artificial Intelligence & Data Branch). The objective is to demystify machine learning by rejecting high-level black-box abstractions and building the core mathematical optimization machinery from first principles.

### Core Constraints & Technical Directives

* **Zero High-Level ML Libraries**: The use of automated estimators or solvers (`scikit-learn`, `statsmodels`, `tensorflow`, `pytorch`) is strictly forbidden. The cost function, partial derivatives, iterative descent loop, convergence criteria, and evaluation metrics are implemented manually.
* **Decoupled Binary Architecture**: Training and inference must exist as independent executables. The training engine computes and serializes the parameters into a lightweight state file (`model.json`), while the inference program consumes these parameters independently.
* **Graceful Degradation & Zero Unhandled Exceptions**: If inference is invoked prior to training (missing `model.json`), the system must degrade gracefully to default parameters ($\theta_0 = 0.0, \theta_1 = 0.0$) without throwing unhandled exceptions. All file I/O, parsing, and user inputs are defensively guarded.
* **Strict Code Quality Standard**: Strict conformance to the Python PEP 8 standard enforced via `flake8` with zero warnings or errors.

---

## 🏗️ Architecture & Modules

The system is decoupled into four modular pipelines: **Data Ingestion**, **Iterative Model Training**, **Interactive Inference**, and **Statistical Evaluation**.

```text
+-----------------------------------------------------------------------------------------------+
|                                        DATA INGESTION                                         |
|                                                                                               |
|   +--------------------------------+                                                          |
|   |   linear_regression_data.csv   |                                                          |
|   +---------------+----------------+                                                          |
|                   |                                                                           |
|                   v                                                                           |
|          +------------------+                                                                 |
|          |    ft_load.py    | ---> Traps I/O faults, validates UTF-8/CSV parsing,             |
|          +--------+---------+      verifies 2D matrix shape (mileage, price)                  |
|                   |                                                                           |
|                   +------------------------------------+                                      |
+-------------------|------------------------------------|--------------------------------------+
                    |                                    |
                    v                                    v
+----------------------------------------+   +----------------------------------------+
|            TRAINING ENGINE             |   |           EVALUATION ENGINE            |
|        (compute_regression.py)         |   |         (compute_precision.py)         |
|                                        |   |                                        |
| 1. Feature Min-Max Normalization:      |   | 1. Evaluates h_θ(x) across dataset     |
|    x_norm = (x - x_min)/(x_max - x_min)|   | 2. Computes Total Sum of Squares:      |
| 2. Vectorized Batch Gradient Descent:  |   |    TSS = Σ (y - y_mean)²               |
|    tmp_0 = α/m * Σ (h_θ(x) - y)        |   | 3. Computes Residual Sum of Squares:   |
|    tmp_1 = α/m * Σ (h_θ(x) - y) * x    |   |    RSS = Σ (y - h_θ(x))²               |
|    Loop until |Δθ_0| & |Δθ_1| < 1e-10  |   | 4. Derives R² Score (% variance):      |
| 3. Analytical Denormalization:         |   |    R² = (1 - RSS / TSS) * 100          |
|    θ_1 = θ_1' / (x_max - x_min)        |   +-------------------+--------------------+
|    θ_0 = θ_0' - θ_1 * x_min            |                       ^
| 4. Model State Serialization           |                       |
+-------------------+--------------------+                       |
                    |                                            |
                    v                                            |
          +--------------------+                                 |
          |     model.json     | --------------------------------+
          |  {θ_0, θ_1 params} |
          +---------+----------+
                    |
                    v
+----------------------------------------+
|            INFERENCE ENGINE            |
|          (estimate_price.py)           |
|                                        |
| 1. Deserializes θ_0, θ_1 from JSON     |
|    (fallback: θ_0 = 0.0, θ_1 = 0.0)    |
| 2. Prompts user for vehicle mileage    |
| 3. Computes h_θ(x) = θ_0 + (θ_1 * x)   |
| 4. Outputs real-time price estimation  |
+----------------------------------------+
```

### Component Breakdown

| Module | Architectural Role | Core Technical Responsibilities |
| :--- | :--- | :--- |
| [`compute_regression.py`](./compute_regression.py) | Training & Optimization | Vectorized batch gradient descent, dynamic min-max feature scaling, convergence monitoring ($\epsilon = 10^{-10}$), analytical parameter denormalization, JSON serialization, and optional Matplotlib plotting (`-p`). |
| [`estimate_price.py`](./estimate_price.py) | Inference Service | Standalone prediction utility: validates numeric CLI input, deserializes model weights with fallback tolerance, and computes the hypothesis $h_\theta(x)$ in $O(1)$ time. |
| [`compute_precision.py`](./compute_precision.py) | Verification & Metrics | Calculates the coefficient of determination ($R^2$) comparing Residual Sum of Squares ($RSS$) against Total Sum of Squares ($TSS$). |
| [`ft_load.py`](./ft_load.py) | Defensive Data Access Layer | Encapsulates CSV ingestion with targeted exception handling (`FileNotFoundError`, `PermissionError`, `ParserError`, `UnicodeDecodeError`, dimensionality verification). |
| [`model.json`](./model.json) | State Persistence Artifact | Lightweight JSON key-value store holding the learned bias ($\theta_0$) and slope ($\theta_1$) parameters. |

---

## 🧠 Engineering Highlights & Learnings

### 1. Vectorized Batch Gradient Descent from First Principles
Rather than relying on closed-form solutions (e.g. the Ordinary Least Squares normal equation $(X^T X)^{-1} X^T y$), the project implements first-order iterative optimization minimizing the Mean Squared Error (MSE) cost function:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

Where the hypothesis is parameterized as:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Simultaneous vectorized parameter updates are executed across all $m$ observations:

$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)$$

$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

Rather than terminating on an arbitrary epoch count, the descent loop tracks relative parameter delta per step, halting only when:

$$\max\left(|\theta_0^{(t)} - \theta_0^{(t-1)}|, |\theta_1^{(t)} - \theta_1^{(t-1)}|\right) < 10^{-10}$$

This guarantees true asymptotic convergence to the global convex minimum.

### 2. Feature Scaling & Analytical Closed-Form Denormalization
* **The Challenge**: The input feature (mileage) spans $22,899 \text{ km} \le x \le 240,000 \text{ km}$ ($O(10^5)$), while prices span $3,650 \le y \le 8,290$ ($O(10^3)$). Operating gradient descent on unscaled features creates severe gradient disparity—the partial derivative with respect to $\theta_1$ is $\approx 10^5$ times larger than with respect to $\theta_0$, leading to divergence, numerical overflow, or requiring an impractically infinitesimal learning rate.
* **The Solution (Min-Max Scaling)**: Inputs are mapped into a balanced unit domain $[0, 1]$:

  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

* **Analytical Denormalization**: Once optimized parameters $(\theta_0', \theta_1')$ are derived on normalized space, computing inference should not require the user or inference engine to know or store dataset statistics ($x_{\min}, x_{\max}$). By algebraic expansion of the linear hypothesis:

  $$h_\theta(x) = \theta_0' + \theta_1' \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right) = \left(\theta_0' - \frac{\theta_1' \cdot x_{\min}}{x_{\max} - x_{\min}}\right) + \left(\frac{\theta_1'}{x_{\max} - x_{\min}}\right) x$$

  The physical parameters are directly denormalized before serialization:

  $$\theta_1 = \frac{\theta_1'}{x_{\max} - x_{\min}}$$

  $$\theta_0 = \theta_0' - \theta_1 \cdot x_{\min}$$

  Consequently, inference evaluates $h_\theta(x) = \theta_0 + \theta_1 x$ directly in raw kilometers with zero runtime transformation overhead.

### 3. Statistical Goodness-of-Fit via $R^2$ Determination
Evaluating performance strictly through Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) provides scale-dependent metrics that do not reveal the predictive power relative to baseline variance. The precision engine computes the scale-invariant **Coefficient of Determination ($R^2$)**:

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_{i=1}^m (y_i - \hat{y}_i)^2}{\sum_{i=1}^m (y_i - \bar{y})^2}$$

On the reference empirical dataset, the model attains:

$$R^2 \approx 73.30\%$$

This validates that mileage alone accounts for approximately $73.30\%$ of the price variation across the fleet, while capturing the intrinsic non-linear vehicle depreciation dynamics without overfitting.

### 4. Defensive Engineering & Graceful Degradation
* **Decoupled Lifecycle**: The inference utility operates completely isolated from the training module.
* **Fallback Safety**: If `model.json` is missing or unreadable, `estimate_price.py` falls back to $(\theta_0 = 0.0, \theta_1 = 0.0)$, informing the user cleanly without panicking.
* **Defensive Ingestion**: `ft_load.py` isolates file system operations, verifying permissions, encoding, CSV integrity, and array dimensionality ($N \times 2$) before returning control to the training script.
* **Code Standard**: All scripts pass `flake8` with 0 warnings under standard PEP 8 limits (79 characters max line length, explicit imports, clean namespace).

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/r-richardcanavaggio/ft_linear_regression.git
cd ft_linear_regression

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install runtime and linting dependencies
pip install numpy pandas matplotlib flake8
```

### 2. Static Analysis & Linter Verification

Verify strict PEP 8 compliance across all project files:

```bash
flake8 compute_precision.py compute_regression.py estimate_price.py ft_load.py
```
*(Expected: exit code 0, zero output)*

### 3. Resetting / Cleaning State

To wipe cached bytecode and reset trained weights:

```bash
rm -rf __pycache__ model.json
```

---

## 💻 Showcase

### 1. Training the Model (`compute_regression.py`)
Trains the model from scratch on `linear_regression_data.csv`, iterating until convergence, denormalizing thetas, and writing to `model.json`:

```bash
python compute_regression.py
```

```text
Data loaded successfully.
Theta 0: 8008.439831886850698 | Theta 1: -4656.591442716205165
Successfully saved new results to 'model.json'
```

*(Optional: append `--plot` or `-p` to display the scatter plot with the fitted regression line via Matplotlib).*

### 2. Interactive Inference (`estimate_price.py`)
Queries the trained parameters to predict the valuation for an entered vehicle mileage:

```bash
python estimate_price.py
```

```text
Enter mileage: 100000
Estimated price for a car with 100000 km is: 6354.703290715405
```

### 3. Model Precision & Goodness-of-Fit (`compute_precision.py`)
Computes the variance explained by the model against the ground truth dataset:

```bash
python compute_precision.py
```

```text
The regression model explains 73.30% of the variance in price.
```

### 4. Fault-Tolerant Fallback Execution
Executing inference when no model has been trained (`model.json` absent):

```bash
rm -f model.json
python estimate_price.py
```

```text
Fichier model.json introuvable, lancement avec valeur par defaut (0,0)
Enter mileage: 100000
Estimated price for a car with 100000 km is: 0.0
```
