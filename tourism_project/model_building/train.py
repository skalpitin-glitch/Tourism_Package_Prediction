# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Setting up MLOps URL and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Defining the path and train and test datasets from HF Space
Xtrain_path = "hf://datasets/skalpitin/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/skalpitin/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/skalpitin/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/skalpitin/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',                          # Age of the customer
    'NumberOfPersonVisiting',       # Total number of people accompanying the customer on the trip.
    'PreferredPropertyStar',        # Preferred hotel rating by the customer.
    'NumberOfTrips',                # Average number of trips the customer takes annually.
    'Passport',                     # Whether the customer holds a valid passport (0: No, 1: Yes).
    'OwnCar',                        # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting',     # Number of children below age 5 accompanying the customer.
    'MonthlyIncome',                # Gross monthly income of the customer.
    'PitchSatisfactionScore',       # Score indicating the customer's satisfaction with the sales pitch.
    'NumberOfFollowups',            # Total number of follow-ups by the salesperson after the sales pitch.
    'DurationOfPitch'               # Duration of the sales pitch delivered to the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'CityTier',              # City category based on development, population, and living standards
    'Occupation',            # Customer's occupation
    'Gender',                # Gender of the customer
    'MaritalStatus',         # Marital status of the customer
    'Designation',           # Customer's designation in their current organization.
    'ProductPitched'         # The type of product pitched to the customer.
]

# Setting the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Defining the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Defining base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Defining hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Starting MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Setting up the classification threshold
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Saving the model locally
    model_path = "Tourism-Package-Prediction_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Defining the HF Space to store the model
    repo_id = "skalpitin/Tourism-Package-Prediction"
    repo_type = "model"

    # Checking if the space exists, if not then create it else use it
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # uploading the model to HF Space
    api.upload_file(
        path_or_fileobj="Tourism-Package-Prediction_v1.joblib",
        path_in_repo="Tourism-Package-Prediction_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
