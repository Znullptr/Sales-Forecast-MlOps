
## 📊 Problem Overview

The original dataset was unstructured and scattered across multiple Word documents. This required:

- 📌 **Data cleansing and transformation**
- 📌 **Normalization and unification** for analysis-ready formatting

---

# Workflow
<p>
   <img width="1000" height="500" src="https://github.com/Znullptr/Sales-Forecast-MlOps/blob/main/workflow.png">
    </p>  
    
# 🧠 Multivariate Sales Forecasting with Full MLOps Pipeline


## 🔍 Time Series Analysis

We performed a full time series analysis, including:

- 📈 **Stationarity testing**
- 🔁 **Trend and seasonality detection**

We experimented with multiple **Machine Learning** and **Statistical models**:

- **Support Vector Machine (SVM)**
- **Linear Regression**
- **Vector Auto Regression (VAR)**

---

## 🚀 MLOps Pipeline Implementation

Once the optimal model was selected, we built and deployed a **full MLOps pipeline**, as illustrated in the diagram.

### 🔄 Data Versioning & Experiment Tracking
- [`DVC`](https://dvc.org/)
- [`MLflow`](https://mlflow.org/)
- [`DAGsHub`](https://dagshub.com/)

### ✅ Model Testing & Validation
- [`Deepchecks`](https://deepchecks.com/)
- `Pytest`

### 🧩 API Development & Deployment
- `FastAPI`
- `Docker`
- `Amazon ECR`
- `AWS ECS`
- `Elastic Load Balancer (ELB)`

### 📡 Monitoring & Drift Detection
- [`Arize AI`](https://arize.com/)

### 📊 Visualization & Reporting
- `Streamlit`
- `GitHub Pages`

### ⚙️ CI/CD Automation
- `GitHub Actions`

---

## ✅ Outcomes

This pipeline enabled:
- ⚡ **Scalable deployment**
- 🛠️ **Continuous monitoring**
- 📦 **Reproducibility & transparency**
- 🔁 **Rapid model iteration**

---

## 📎 Tools & Technologies

| Category                 | Tools Used                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| Data Handling            | Python, Pandas, NumPy                                                      |
| Modeling                 | SVM, Linear Regression, VAR                                                |
| MLOps & Tracking         | DVC, MLflow, DAGsHub                                                       |
| Testing & Validation     | Deepchecks, Pytest                                                         |
| Deployment               | FastAPI, Docker, AWS ECS, ELB                                              |
| Monitoring               | Arize AI                                                                   |
| Visualization & Reports | Streamlit, GitHub Pages                                                    |
| CI/CD                    | GitHub Actions                                                             |

---

## 📌 Note

For a deeper look into the architecture and code, please refer to the `/notebooks`, `/main`, and `/data` directories in this repo.

---
