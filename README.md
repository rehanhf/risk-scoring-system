### Credit Risk Scoring System

This system is an end-to-end infrastructure designed to predict the probability of credit default and automate loan approval decisions based on economic utility. It prioritizes financial loss mitigation over abstract model accuracy.

---

### 1. The Core Logic: Financial Decisioning
Unlike standard machine learning projects, this system does not care about "accuracy." It cares about **Expected Value**.

*   **The Problem**: Missing a default (False Negative) costs the bank **$5,000** in lost principal. Wrongly rejecting a good customer (False Positive) only costs **$500** in missed interest.
*   **The Solution**: We set the decision threshold at **0.28**. If the model is only 29% sure a customer will default, we reject them. This "paranoid" setting minimizes the total money lost.

---

### 2. System Architecture
The pipeline is strictly modular to ensure that the logic used to train the model is identical to the logic used when a customer applies via the API.

1.  **Data Ingestion**: Programmatic download and conversion of raw financial records.
2.  **Temporal Splitting**: Data is split by time, not randomly. We train on the past to predict the future. This prevents "looking ahead" at future economic conditions.
3.  **The Transformer**: A serialized `ColumnTransformer` that handles missing values and converts categories into mathematical vectors. It is fitted only on training data to prevent leakage.
4.  **The Engine**: A LightGBM Gradient Boosting model. It was chosen over Deep Learning (PyTorch) because it is faster, uses 95% less RAM, and is easier for financial regulators to audit.
5.  **The API**: A FastAPI service wrapped in a Docker container. It enforces strict data types (e.g., Age must be $\ge 18$).

---

### 3. Setup & Installation

**Prerequisites**: `uv` (Python manager), `Docker`.

1.  **Environment Setup**:
    ```powershell
    uv sync
    ```
2.  **Run Pipeline (Train Model)**:
    ```powershell
    python src/data/ingest.py
    python src/data/split.py
    python src/features/preprocess.py
    python src/training/train_lgbm.py
    ```
3.  **Deploy via Docker**:
    ```powershell
    docker build -t risk-engine:v2 -f api/Dockerfile .
    docker run -d -p 8000:8000 --name risk-container risk-engine:v2
    ```

---

### 4. Interactive Demo (The Flow)
To visualize how a loan officer interacts with the system:

1.  **Start the UI**:
    ```powershell
    uv run streamlit run ui/app.py
    ```
2.  **The Approval Flow**:
    *   Input a "Safe" profile: High credit limit, perfect repayment history (PAY = -1).
    *   The API calculates a low probability ($< 0.28$).
    *   **Result**: `APPROVE`.
3.  **The Rejection Flow**:
    *   Input a "Risky" profile: History of late payments (PAY > 1), zero payment amounts.
    *   The API calculates a high probability ($\ge 0.28$).
    *   **Result**: `REJECT`.
    ![DEMO](/demo/image.png)

---

### 5. Monitoring & Reliability
Models decay as the economy changes. This system includes a **Monitoring Layer**:
*   **PSI (Population Stability Index)**: Measures if current applicants look different from the original training group.
*   **Threshold**: A PSI $> 0.25$ triggers an automatic alert, indicating the model is no longer reliable and must be retrained on new data.

---

### 6. Technical Integrity
*   **Strict Typing**: Pydantic schemas block malformed data at the door.
*   **Static Analysis**: `Ruff` and `Mypy` ensure code quality before every commit.
*   **Explainability**: SHAP (Shapley Values) explains *why* a specific person was rejected, ensuring compliance with "Right to Explanation" laws.

---

### 7. Mathematical Foundation & Key Equations

The system operates on three primary mathematical pillars:

**A. Expected Value (Decision Objective)**
The decision threshold is not arbitrary; it minimizes the Expected Loss ($EL$).
$$EL(p, t) = \begin{cases} p \times C_{FN} & \text{if } p < t \text{ (Approve, but defaults)} \\ (1-p) \times C_{FP} & \text{if } p \ge t \text{ (Reject, but would stay good)} \end{cases}$$
Where $p$ is the model's raw probability and $t$ is the optimized threshold (0.28).

**B. Population Stability Index (PSI)**
Used to detect "Model Decay" or "Data Drift."
$$PSI = \sum_{i=1}^{B} (\%Actual_i - \%Expected_i) \times \ln\left(\frac{\%Actual_i}{\%Expected_i}\right)$$
*   $Actual$: New production data distribution.
*   $Expected$: Training data distribution.
*   $B$: Number of bins.

**C. Weighted Binary Cross-Entropy (Training Loss)**
Used to force the model to learn from the minority "Default" class.
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} [w \cdot y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$
Where $w = 3.38$ (scale_pos_weight).

---

### 8. Known Weaknesses & Mitigation Strategies

| Weakness | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Fixed Threshold** | Static $t=0.28$ assumes interest rates and bank liquidity never change. | Implement **Dynamic Thresholding** based on the bank’s monthly cost-of-capital. |
| **Tabular Bias** | Model cannot see the "Social Graph" (e.g., a group of fraudsters applying together). | Future integration of **Graph Neural Networks (GNNs)** to detect ring behavior. |
| **Feature Staleness** | Credit history (`PAY_0`) is updated monthly; it misses intra-month "bust-out" fraud. | Implement **Real-time Streaming Features** (Flink/Spark) for transaction velocity. |
| **Uncalibrated Scores** | `scale_pos_weight` distorts true probabilities, making scores look "scarier" than they are. | Apply **Isotonic Regression** or **Platt Scaling** after the LightGBM head. |

---

### 9. Alternative Methods Explored

*   **Logistic Regression**: Rejected due to inability to capture non-linear interactions between `LIMIT_BAL` and `AGE` without manual feature engineering.
*   **XGBoost**: Performance was comparable to LightGBM, but LightGBM was selected for significantly faster training and lower memory usage during the SHAP computation phase.
*   **TabNet**: Evaluated for deep learning on tabular data. Rejected because the performance gain ($+0.002$ PR-AUC) did not justify the $10\times$ increase in inference latency and GPU requirement.

---

### 10. Future Development Roadmap (Phase 13+)

1.  **Automated Retraining (CI/CD/CT)**: Trigger a new training job automatically when PSI exceeds 0.20 for 3 consecutive days.
2.  **Challenger Models**: Deploy a "Shadow Model" (e.g., the PyTorch MLP) in the background to compare its real-world performance against the LightGBM "Champion."
3.  **Adversarial Robustness**: Implement defensive testing against "adversarial noise"—payloads designed to trick the model into an `APPROVE` decision by slightly altering `BILL_AMT` values.

---

### 11. Key Code Snippet: The "Gatekeeper"

This is the most critical block in the system (`api/main.py`), where math meets business authority:

```python
# The Economic Gatekeeper
decision = "REJECT" if prob >= 0.28 else "APPROVE"

# Risk Tiering for Granular Pricing
if prob < 0.10: tier = "LOW"             # Prime
elif prob < 0.28: tier = "MEDIUM"        # Near-Prime
elif prob < 0.60: tier = "HIGH"          # Sub-Prime
else: tier = "CRITICAL"                  # Automatic Decline
```

---
