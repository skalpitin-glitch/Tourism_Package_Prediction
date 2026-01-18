# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Defining constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/skalpitin/Tourism-Package-Prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)

# Data cleaning steps

## 1 - Dropping the Serial Number column as it has only unique values for each row.
tourism_dataset = tourism_dataset.drop(tourism_dataset.columns[0], axis=1)

## 2 - Dropping the CustomerID column as it has only unique values for each row. 
tourism_dataset.drop(columns=['CustomerID'], inplace=True)

## 3 - Correcting the incorrect values in Gender column ("Fe Male" should be corrected to "Female")
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe Male', 'Female')

# Defining the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                          # Age of the customer
    'NumberOfPersonVisiting',       # Total number of people accompanying the customer on the trip.
    'PreferredPropertyStar',        # Preferred hotel rating by the customer.
    'NumberOfTrips',                # Average number of trips the customer takes annually.
    'Passport',                     # Whether the customer holds a valid passport (0: No, 1: Yes).
    'OwnCar',                       # Whether the customer owns a car (0: No, 1: Yes).
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

# Defining predictor matrix (X) using selected numeric and categorical features.  
X = tourism_dataset[numeric_features + categorical_features]

# Defining target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

# Writing the 4 datasets into the csv files into the local folder. 
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Uploading the datasets to HF space from the local folder. 
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="skalpitin/Tourism-Package-Prediction",
        repo_type="dataset",
    )
