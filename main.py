import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Set MLflow Experiment
mlflow.set_experiment("Breast Cancer Classification")

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log parameters
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        if hasattr(model, "kernel"):
            mlflow.log_param("kernel", model.kernel)

        if hasattr(model, "n_neighbors"):
            mlflow.log_param("n_neighbors", model.n_neighbors)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name}")
        print(f" Accuracy: {accuracy}")
        print(f" Precision: {precision}")
        print(f" Recall: {recall}")
        print(f" F1 Score: {f1}")
        print("-" * 30)