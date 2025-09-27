# 🎵 Spotify Recommender System

An **end-to-end Machine Learning project** that builds a **music recommender system** using Spotify data.  
This project integrates **DVC**, **MLflow**, **GitHub Actions CI/CD**, **Docker**, and **AWS EC2/ECR** for a complete MLOps pipeline.

---

## 🚀 CI/CD Pipeline

This repository includes a **GitHub Actions workflow** (`.github/workflows/ci-cd.yaml`) that automates the following:

* ✅ Install dependencies and set up environment  
* 🔑 Configure AWS credentials  
* 📊 Configure MLflow with **DagsHub**  
* 📦 Pull datasets and models with **DVC**  
* 🔄 Run the training pipeline (`dvc repro`)  
* 🧪 Test model registry and performance with `pytest`  
* 🏷️ Promote best model to production  
* 🐳 Build & push Docker image to **Amazon ECR**  
* 🚀 Deploy containerized app to **AWS EC2**  

---

## 📂 Project Structure

```
├── LICENSE
├── Makefile           <- Commands for automation (data, train, test, deploy)
├── README.md          <- Project documentation
├── data
│   ├── raw            <- Raw Spotify dataset
│   ├── processed      <- Cleaned and feature-engineered data
│   ├── interim        <- Intermediate datasets
│   └── external       <- External sources (if any)
│
├── models             <- Trained and serialized ML models
├── notebooks          <- Jupyter notebooks for EDA & experimentation
├── reports            <- Generated reports and visualizations
├── src                <- Source code for data processing and ML pipeline
├── tests              <- Unit and integration tests
├── Scripts            <- Deployment and utility scripts (e.g., promote model)
├── app.py             <- Main application entry point (FastAPI/Flask app)
├── requirements.txt   <- Python dependencies
├── setup.py           <- Package setup
├── Dockerfile         <- Docker container definition
├── dvc.yaml           <- DVC pipeline stages
└── .github/workflows  <- CI/CD workflow definitions
```

---

## ⚙️ Typical Workflow

1. **Data Collection** → `data/raw/`  
2. **Data Processing & Cleaning** → `src/`  
3. **EDA** → `notebooks/`  
4. **Feature Engineering** → `src/`  
5. **Model Training** → `dvc repro`  
6. **Model Evaluation** → `reports/`  
7. **Inference / Prediction** → `app.py`  
8. **Deployment** → Docker + AWS EC2  
9. **Testing** → `pytest tests/`  
10. **Continuous Integration** → GitHub Actions CI/CD  

Use **Makefile** for automation:

```bash
make data        # Process data
make train       # Train model
make test        # Run tests
make lint        # Lint code
```

---

## 🛠️ Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd Spotify-Recommender-System

# Setup environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Verify setup
pytest tests/
```

---

## 📊 DVC Pipeline

| Stage              | Description                           | Output                          |
|--------------------|---------------------------------------|---------------------------------|
| `data_load`        | Load raw Spotify dataset              | `data/raw/`                     |
| `data_process`     | Clean & process dataset               | `data/processed/`               |
| `feature_engineer` | Feature engineering for model         | `data/interim/`                 |
| `train`            | Train recommender model               | `models/model.pkl`              |
| `evaluate`         | Evaluate model & log metrics to MLflow| `reports/metrics.json`          |
| `deploy`           | Package model for deployment          | `Docker image / EC2 container`  |

---



## 🙌 Acknowledgments

* Inspired by **Spotify** recommendation engine  
* **DVC, MLflow, DagsHub, AWS, Docker** for MLOps backbone  
