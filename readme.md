# Length-of-Stay Prediction  
The goal of this project is to estimate the length of a patient's hospital stay given the patient's demographics and hospital discourse information. This is achieved through machine learning principles. Two tree-based ensemble algorithms, random forest and XGBoost, were chosen to compare their performances. The random forest algorithm was implemented manually and XGBoost was imported from the XGBoost library. These models were trained on the MIMIC-IV library, an anonymized database of over 500,000 hospital admissions. The hyperparameters were tuned using grid-search and Bayesian optimization methods to maximize accuracy. Both regression and classification models were implemented, with the latter binning the length-of-stay outcome into short (1-2 days), medium (3-5 days), and long (6+ days) stays.

## Project Structure
### Data:
data_extraction.py connects to the BigQuery server and queries it via SQL to retrieve the MIMIC-IV database containing the necessary features. It returns the data in a pandas dataframe.

preprocessing.py modifies the data to ensure it is properly formatted for the machine learning models. First, it separates the data into the input features and the isolated length-of-stay outcome. The ICD codes are abbreviated to only include the chapter and category to avoid scarcity. This primary diagnosis is target encoded rather than one-hot encoded to reduce dimensionality. All other categorical variables are one-hot encoded. If the classify flag is set, the outcome measure is binned as described above. There is an option to cache the outcomes in csv files to avoid repeated BigQuery querying.

### Models:
decision_tree.py contains the manual implementation of the basic decision tree used later in the random forest algorithm. This file consists of nodes representing splits based on feature characteristics. The node class contains the feature type, threshold for the decision, left and right nodes, and a value for leaf nodes to represent the final prediction (otherwise defaulting to None for splitting nodes). It also includes a method to determine if it is a leaf node by checking if value is None. 

The decision tree class contains attributes max_depth and min_sample_split to indicate when the recursion should stop. It also has n_features to clarify how many features are in this specific decision tree and has access to the root node. There is one method to fit the decision tree and one method to predict a value. 

The fit method calls the helper function grow_tree, which builds the tree and returns the root node of the resulting tree. The grow_tree function first checks the stopping criteria (depth >= max_depth, n_samples < min_samples_split, or n_labels == 1). If any of these criteria are met, the leaf node value is calculated as the average of all labels and returned. Otherwise, the best split is found. This is done by calling another helper function, best_split. This takes X, y, and the feature indices used in the decision tree. This function looks at all thresholds and finds the split that maximized information gain. It goes through each feature and finds all unique values to use as thresholds. To find this, it calls another helper function, information_gain. This takes in y, X_column (feature), and threshold, and returns the information gain. This is found by subtracting the weighted average of the variance of the children from the variance of the parent. The children are creating by splitting according to the current threshold in its own helper function. Once the best split is found, the child nodes are created by splitting the samples according to the best threshold. Finally, the grow_tree helper is recursively called on the child nodes to continue splitting or stop as leaf nodes.

The predict method calls the helper function traverse_tree, which takes a value and the current node it is at. If the current node is a leaf node, it returns the value. Otherwise, it compares the x value to the node’s threshold, and recursively calls itself on the correct child node.

random_forest.py is the manual implementation of the ensemble algorithm. The random forest is initialized with attributes for n_trees, max_depth, min_samples_split, n_features, and an array to hold the trees. Similar to the decision tree class, random forest also has a fit and predict function.

The fit method creates n_trees decision trees. While each decision tree is being made, they are also being fitted according to a sample of the overall training dataset. The sample is created using the helper function bootstrap_samples, which randomly selects samples from the training dataset with replacement. 

The predict method iterates through each tree and stores the predicted values in an array. The average value for each sample is calculated and the predictions for each sample is returned as an array.

The train_random_forest.py and train_xgboost.py files train the algorithms on the preprocessed data. train_random_forest.py imports the random forest and decision tree classes described above, whereas xgboost imports the algorithm from the xgboost library. Both training functions take the features dataframe, outcome series, and classification flag as inputs and print the evaluation metrics of the trained algorithm.

### Tests:
There are test files for the corresponding files within this directory. These contain unit tests for the functions and classes and can be run with the command `python -B -m unittest discover`.

### Tuning:
tune_xgboost.py provides a grid-search to determine the best hyperparameters. There are four hyperparameters with two options each. This does not search the entire decision space but provides a rudimentary ability to optimize the model.

tune_xgboost_optuna.py implements Bayesion optimization more throroughly find the best hyperparameters. This involves strategically building a probability model of the object function to evaluate the true objective function. This specific instance runs 50 studies but can be increased to further derive the optimal hyperparameters.

## Alternative Options and Other Considerations
Several traditional regression models were considered but ultimately not selected due to limitations in scalability, interpretability, or suitability for the dataset:

Linear, Ridge, Lasso, and Stochastic Gradient Descent (SGD) Regression
These models offer high interpretability. Linear regression directly relates each feature to the target via its coefficient. Ridge and Lasso extend this by adding regularization: Ridge penalizes large coefficients to reduce overfitting (L2 penalty), while Lasso also performs feature selection by driving some coefficients to zero (L1 penalty). Stochastic Gradient Descent (SGD) is well-suited for very large datasets or streaming data. However, these linear models assume a straight-line relationship between features and outcomes, which is often too simplistic for complex medical data like hospital length of stay. Additionally, SGD is sensitive to feature scaling and requires careful hyperparameter tuning, making it less ideal for early-stage modeling.

Support Vector Regression (SVR)
SVR introduces a tolerance margin (epsilon) around the optimal line and only optimizes based on errors outside this range, helping to reduce overfitting. With kernel functions, it can model non-linear relationships as well. However, it is computationally expensive, scales poorly with large datasets, and has lower interpretability, making it a less practical choice for this application.

K-Nearest Neighbors (KNN)
KNN regression predicts outcomes by averaging target values of the nearest neighbors. It requires no training time and is intuitive, but suffers from major scalability issues and memory inefficiency. It is also highly sensitive to irrelevant features and performs poorly in high-dimensional spaces—challenges that are common in clinical datasets.

Tree-based methods were preferred for their ability to capture non-linear interactions and robustness to feature scaling.

## How to Run Locally
### 1. Clone the Repository:
```
git clone https://github.com/yourusername/los-predictor.git
cd los-predictor
```
### 2. Create and Activate a Virtual Environment
```
python -m venv .venv
source .venv/bin/activate (MacOS and Linux)
.venv\Scripts\activate (Windows)
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Set Up Environmental Variables
```
GCLOUD_PROJECT=your-gcloud-project-id
```
Make sure your local environment is authenticated with Google Cloud SDK: `gcloud auth application-default login`

### 5. Fetch and Preprocess Data
```
python data/load_pipeline.py
```
This will either fetch the data from BigQuery and preprocess it or use cached clean_output.csv and target.csv if from_cache=True.

### 6. Train Models
```
python models/train_random_forest.py # Custom implementation
python models/train_xgboost.py # Library-based implementation
```
Train either model depending on your use case

### 7. Tune Models (Optional)
```
python tuning/tune_xgboost.py # Grid Search
python tuning/tune_xgboost_optuna.py # Bayesian Optimization
```

## Input Data
The following features were extracted from the MIMIC-IV database to train the models:

- subject_id: Unique patient identifier

- hadm_id: Unique hospital admission identifier

- gender: Patient gender (M or F)

- age: Patient age at the time of admission

- race: Reported race of the patient

- admission_type: Admission category (e.g., Emergency, Elective)

- insurance: Type of insurance coverage

- admittime: Timestamp of hospital admission

- dischtime: Timestamp of discharge

- primary_diagnosis: ICD-10 diagnosis code for the stay

- length_of_stay: Calculated as the number of days between admission and discharge

Note: Access to MIMIC-IV requires credentialing through PhysioNet, which involves completing a CITI module and providing a reason for access.

## Known Issues
The current implementation of the random forest runs very slowly. Its default setting creates 10 decision trees. Each decision tree may take up to 30 minutes to train on the ~500,000 records for a total training time of 5 hours.

## Future Improvements
The predictive power of the features used in this project is lacking, with a final classification accuracy of 67%. More features should be incorporated to allow for more accurate outcomes. Specifically, secondary diagnoses and comorbidities may provide enough context to greatly improve the model predictions.

## Data Source & Acknowledgements
This project uses data from the MIMIC-IV (Medical Information Mart for Intensive Care IV) database.

Access to MIMIC-IV is provided through PhysioNet.

Use of the data is subject to the terms and conditions outlined by PhysioNet and requires completion of human subjects research training.

Johnson AE, Pollard TJ, Shen L, et al. MIMIC-IV (version 1.0). PhysioNet. 2020.
https://doi.org/10.13026/a3wn-hq05
