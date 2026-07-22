# Customer Churn Prediction using Logistic Regression

## Objective
Predict whether a telecom customer is likely to churn using demographic and service-usage information from the Telco Customer Churn dataset.

## Dataset Link
- [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Files in this Folder
- `Assignment-2.py` - runnable solution for all assignment tasks
- `Assignment-2.ipynb` - notebook version of the assignment
- `requirements.txt` - Python dependencies

## Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `kagglehub`

## Methodology
1. Load the dataset with Pandas and inspect the first five rows.
2. Identify numerical features, categorical features, and the target variable (`Churn`).
3. Check missing values and handle them with imputation.
4. Encode categorical features with one-hot encoding and split the data into 80% training and 20% testing.
5. Train a Logistic Regression model using a preprocessing pipeline.
6. Evaluate the model with Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Results
The script can load the CSV from this folder or download it automatically with `kagglehub`.

## Conclusion
Logistic Regression is a strong baseline for churn prediction because it is simple, fast, and easy to interpret. In this problem, churn is typically influenced by factors such as tenure, contract type, monthly charges, and service usage patterns. A limitation of Logistic Regression is that it assumes a largely linear relationship between features and the target, so it may miss complex nonlinear interactions in customer behavior.

## Note
Do not upload the dataset to GitHub. Keep only the code and documentation in the repository, and include the Kaggle link instead.