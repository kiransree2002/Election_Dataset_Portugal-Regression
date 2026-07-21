## Project Overview: Election Dataset Portugal - Regression

This notebook details a machine learning project focused on predicting `FinalMandates` using the Real-Time Election Results Portugal 2019 dataset. The process involves comprehensive data preprocessing, feature engineering, and the implementation and evaluation of various regression models, particularly focusing on the K-Nearest Neighbors (KNN) Regressor.

### Objectives

- Understand the dataset structure.
- Identify and handle missing or inconsistent values.
- Detect and address duplicate records.
- Analyze the distribution of numerical and categorical variables.
- Identify and manage outliers.
- Prepare the dataset for statistical analysis and machine learning.
- Build and evaluate regression models to predict `FinalMandates`.

### Preprocessing Steps Performed

#### 1. Data Loading & Initial Exploration
- Dataset loaded from `/content/drive/MyDrive/ICT - Ai Ml/DataPreprocessing - Group 1/Data/ElectionData.csv`.
- Initial checks performed using `head()`, `shape`, `info()`, and `describe()` to understand data types, dimensions, and basic statistics.

#### 2. Missing Value & Duplicate Handling
- **Missing Values**: Checked using `isnull().sum()`. *No missing values were found*.
- **Duplicate Records**: Checked using `duplicated().sum()`. *No duplicate records were found*.

#### 3. Outlier Detection & Handling
- Outliers were identified using **Boxplots** for numerical features.
- **IQR method** was applied to quantify outliers for each numerical column.
- *Decision*: Outliers were identified but **not removed** or clipped, as they were deemed potentially representative of real-world variations in election data (e.g., large vote counts in major territories/parties).

#### 4. Data Visualization (EDA)
- **Countplot**: Visualized the distribution of political parties (`Party`).
- **Histograms**: Displayed distributions of numerical features to understand their spread and skewness.
- **Boxplots**: Used to visualize the distribution and identify outliers in numerical features.
- **Scatterplots**: Plotted relationships between numerical features and the target variable (`FinalMandates`) to understand correlations.

#### 5. Feature Engineering
- **Correlation Analysis**: Calculated Pearson correlation matrix for numerical features.
- **Correlation Heatmap**: Visualized correlations to identify highly correlated features.
- **Mutual Information**: Calculated Mutual Information (MI) scores between features and `FinalMandates` to determine feature importance.
- **Feature Removal**: 
    - Highly correlated features (`Hondt`, `Mandates`) were removed to reduce multicollinearity.
    - Low mutual information and low correlation features (`TimeElapsed`, `blankVotesPercentage`, `pre.blankVotesPercentage`, `nullVotesPercentage`, `pre.nullVotesPercentage`) were removed.
    - The `time` column was dropped due to high cardinality and its limited relevance to election outcomes.

#### 6. Feature Scaling
- **Standard Scaling**: `StandardScaler` was applied to all remaining numerical features (both `int64` and `float64` types) to standardize their ranges (mean=0, standard deviation=1).

#### 7. Encoding Categorical Columns
- **One-Hot Encoding**: Categorical features (`territoryName`, `Party`) were converted into numerical format using `OneHotEncoder(sparse_output=False)`.

### Model Building

#### 1. Train-Test Split
- The preprocessed data `X` (features) and `y` (target: `FinalMandates`) were split into training and testing sets with a `test_size` of 0.2 and `random_state=33`.

#### 2. K-Nearest Neighbors (KNN) Regressor
- **Baseline KNN Model**: Trained a `KNeighborsRegressor` on the `X_train` and `y_train`.
- **Boosting with KNN**: An `AdaBoostRegressor` was implemented using `KNeighborsRegressor` as the base estimator.
- **Cross-Validation (K-Fold)**: `KFold` cross-validation with 4 splits (`n_splits=4`, `shuffle=True`, `random_state=33`) was applied to the `KNeighborsRegressor`.
- **Hyperparameter Tuning (Grid Search CV)**: `GridSearchCV` was used to find optimal hyperparameters (`n_neighbors`, `weights`) for `KNeighborsRegressor`.
- **Hyperparameter Tuning (Randomized Search CV)**: `RandomizedSearchCV` was used for a broader search of hyperparameters (`n_neighbors`, `weights`, `p`) for `KNeighborsRegressor`.

### Model Evaluation

The models were evaluated using the following regression metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

#### KNN Regressor Evaluation:
- **Baseline KNN Model**:
    - MSE: 0.1363
    - MAE: 0.0236
    - R² Score: 0.9974
- **Boosting KNN Regression Model**:
    - MSE: 0.0002
    - MAE: 0.0006
    - R² Score: 0.99999
- **KFold Cross-validation for KNN Regressor**:
    - Cross validation scores: [99.75%, 99.96%, 99.88%, 99.74%]
    - Mean of CV scores: 99.83%
- **KNN Regressor (After Hyperparameter Tuning - Grid Search CV)**:
    - Best Parameters: `{'n_neighbors': 3, 'weights': 'distance'}`
    - Best CV Score: 0.9983
    - MAE: 0.0023
    - MSE: 0.0160
    - R² Score: 0.9997
- **KNN Regressor (Randomized Search CV)**:
    - Best Parameters: `{'weights': 'distance', 'p': 1, 'n_neighbors': 3}`
    - Best CV Score: 0.9983
    - MAE: 0.0088
    - MSE: 0.0261
    - R² Score: 0.9995

*(Further models like SVM, Decision Tree, and Linear Regression are building....)*
