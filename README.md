# 🧠 MLflow Multi-Model Classification Demo

This project demonstrates how to use **MLflow** to track, manage, and compare multiple machine learning models on a classification problem.

Instead of training just one model, this project trains **five different machine learning algorithms**, evaluates their performance using standard metrics, and logs everything using **MLflow** for visualization and reproducibility.

---

## 🎯 Project Objective

The objectives of this project are:

✔ Train multiple ML models  
✔ Evaluate model performance  
✔ Track experiments using MLflow  
✔ Compare models visually  
✔ Understand experiment tracking (MLOps concept)

---

## 📊 Problem Statement

We use the **Breast Cancer Wisconsin Dataset** from Scikit-learn.

**Task:**  
Predict whether a tumor is:

- **Malignant** (Cancerous)
- **Benign** (Non-cancerous)

This is a **binary classification problem**.

---

## 🤖 Machine Learning Models Implemented

Five different models are trained and compared:

1️⃣ Logistic Regression  
2️⃣ Decision Tree Classifier  
3️⃣ Random Forest Classifier  
4️⃣ Support Vector Machine (SVM)  
5️⃣ K-Nearest Neighbors (KNN)

Each model is:

✔ Trained  
✔ Evaluated  
✔ Logged into MLflow  

---

## 📈 Evaluation Metrics

For each model, we log:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics help measure classification performance.

---

## ⚙️ Technologies & Libraries Used

- Python 3.10
- Scikit-learn
- MLflow
- NumPy
- Conda

---

## 🏗️ Project Structure

```
MLflow-MultiModel-Demo/
│
├── main.py              # Main script (training + MLflow logging)
├── requirements.txt     # Dependencies
├── README.md            # Documentation
└── mlruns/              # MLflow tracking data (auto-generated)
```

---

## 🚀 Setup Instructions (Windows + Conda)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/srinivasguptha81/MLWorkflow.git
```

---

### 2️⃣ Create Conda Environment

```bash
conda create -n mlflow_env python=3.10
```

---

### 3️⃣ Activate Environment

```bash
conda activate mlflow_env
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

(Optional if setuptools warning appears)

```bash
pip install "setuptools<81"
```

---

## ▶️ Running the Project

```bash
python main.py
```

This will:

✔ Load dataset  
✔ Train five models  
✔ Compute metrics  
✔ Log experiments into MLflow  

---

## 📂 MLflow Tracking

After running the script, MLflow creates:

```
mlruns/
```

This folder contains:

✔ Parameters  
✔ Metrics  
✔ Artifacts (trained models)  
✔ Run metadata  

---

## 🌐 Launch MLflow UI

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 MLflow Dashboard Features

Inside MLflow UI:

✔ View experiment runs  
✔ Compare model metrics  
✔ Inspect parameters  
✔ Download saved models  

This helps identify:

🏆 Best model  
📉 Worst model  

---

## 💡 Why MLflow is Important

In machine learning projects, models are trained multiple times with:

- Different parameters  
- Different algorithms  
- Different datasets  

Without MLflow:

❌ Hard to track experiments  
❌ Difficult to reproduce results  
❌ Disorganized workflow  

With MLflow:

✅ Structured tracking  
✅ Easy comparison  
✅ Reproducibility  
✅ Model management  

---

## 🧠 Key Learnings

✔ Multi-model experimentation  
✔ Classification metrics evaluation  
✔ Experiment tracking  
✔ Reproducible ML workflow  
✔ Introduction to MLOps  

---

## 📝 Project Abstract

This project demonstrates MLflow-based experiment tracking by training and comparing five classification algorithms: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and K-Nearest Neighbors. The Breast Cancer dataset is used for binary classification. Performance metrics including Accuracy, Precision, Recall, and F1 Score are logged into MLflow. The project highlights reproducibility, experiment management, and model comparison.

---

## 🏁 Conclusion

This project provides practical exposure to:

👉 Machine Learning Model Comparison  
👉 Experiment Tracking with MLflow  
👉 Reproducible ML Pipelines  

It simulates how ML experiments are managed in real-world production environments.

---

