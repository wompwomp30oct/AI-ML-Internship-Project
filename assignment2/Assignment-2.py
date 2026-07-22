"""Customer Churn Prediction using Logistic Regression

Assignment 2 - AIML

This script completes the assignment tasks for the Telco Customer Churn dataset.
Place the CSV file in the same folder as this script before running it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import kagglehub
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)


def load_dataset() -> pd.DataFrame:
    """Load the Telco churn dataset from the current folder."""
    candidates = [
        Path("Telco-Customer-Churn.csv"),
        Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    dataset_dir = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
    csv_candidates = sorted(dataset_dir.rglob("*.csv"))

    if not csv_candidates:
        raise FileNotFoundError(
            "Kaggle dataset was downloaded, but no CSV file was found inside the extracted folder."
        )

    return pd.read_csv(csv_candidates[0])


def main() -> None:
    # Task 1: Data Understanding
    data = load_dataset()
    print("First five records:\n")
    print(data.head())

    numerical_features_raw = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features_raw = data.select_dtypes(include=["object"]).columns.tolist()
    target_variable = "Churn"

    print("\nTask 1: Feature identification")
    print("Numerical features:", numerical_features_raw)
    print("Categorical features:", categorical_features_raw)
    print("Target variable:", target_variable)

    # Task 2: Data Preprocessing
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("\nTask 2: Missing values")
    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    x = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"]

    numerical_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()

    print("\nPost-cleaning numerical features:", numerical_features)
    print("Post-cleaning categorical features:", categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Task 3: Model Development
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Task 4: Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTask 4: Evaluation metrics")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No Churn", "Predicted Churn"],
        yticklabels=["Actual No Churn", "Actual Churn"],
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

    # Task 4: Observations
    observations = [
        "Accuracy can look strong because the churn dataset is usually imbalanced toward non-churn customers.",
        "Recall is especially important here because false negatives mean the model misses customers who are likely to leave.",
        "The confusion matrix helps reveal whether the model is better at identifying non-churners than churners.",
    ]

    print("\nObservations:")
    for index, observation in enumerate(observations, start=1):
        print(f"{index}. {observation}")

    # Task 5: Conclusion
    conclusion = (
        "The logistic regression model provides a clear baseline for customer churn prediction "
        "by linking churn to demographic and service-use patterns in the Telco dataset. "
        "In practice, customers with shorter tenure, higher monthly charges, month-to-month contracts, "
        "and fewer add-on services often show a higher churn risk. The model is easy to interpret and "
        "works well as a first benchmark, especially when combined with careful preprocessing. "
        "Its main limitation is that it learns a largely linear decision boundary, so it may miss complex "
        "nonlinear interactions that influence churn. As a result, more advanced tree-based or ensemble methods "
        "may improve recall for churn customers."
    )

    print("\nConclusion:\n")
    print(conclusion)


if __name__ == "__main__":
    main()
