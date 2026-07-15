# Data Preprocessing README

This document outlines the steps taken to preprocess the `ElectionData.csv` dataset, preparing it for machine learning model development. The goal of this preprocessing is to clean the data, handle outliers, perform feature selection, scale numerical features, and encode categorical features.

## 1. Data Loading and Initial Exploration

- The `ElectionData.csv` file was loaded into a pandas DataFrame named `df_election`.
- Initial exploration included checking data types (`df_election.info()`), dimensions (`df_election.shape`), descriptive statistics (`df_election.describe()`), and identifying missing values (`df_election.isnull().sum()`) and duplicate rows (`df_election.duplicated().sum()`).
- No missing values or duplicate rows were found in the initial dataset.
- Numerical and categorical columns were identified using `select_dtypes`.

## 2. Exploratory Data Analysis (EDA) Highlights

- Count plots were generated for the `Party` column to visualize the distribution of political parties.
- Histograms were plotted for all numerical columns to understand their distributions.
- Boxplots were generated for all numerical columns to visually identify potential outliers.
- Scatterplots were generated for pairs of numerical columns to observe relationships.

## 3. Outlier Handling

Outliers were detected using the Interquartile Range (IQR) method. A two-pronged approach was used for handling:

- **Clipping**: For specific columns (`'Votes'`, `'validVotesPercentage'`, `'Percentage'`, `'totalVoters'`, `'votersPercentage'`, `'blankVotes'`, `'availableMandates'`), outliers were clipped to the lower and upper bounds defined by the IQR method. This prevents extreme values from unduly influencing the model while retaining the data points.
- **Removal**: For all other numerical columns, rows containing outliers were removed from the `df_election` DataFrame. This was done to ensure a clean dataset for subsequent analysis and modeling.

## 4. Feature Engineering and Selection

### a. High Correlation Feature Removal

- A correlation matrix was calculated for all numerical features.
- A heatmap was visualized to show inter-feature correlations.
- Features with an absolute correlation greater than 0.90 with other features were identified and removed to reduce multicollinearity, which can negatively impact model performance and interpretability. The removed features were: `['nullVotes', 'subscribedVoters', 'totalVoters', 'pre.blankVotes', 'pre.blankVotesPercentage', 'pre.nullVotes', 'pre.nullVotesPercentage', 'pre.subscribedVoters', 'pre.totalVoters', 'validVotesPercentage']`.

### b. Mutual Information Feature Selection

- Mutual Information (MI) scores were calculated to quantify the dependency between each feature and the target variable (`FinalMandates`).
- Features with an MI score less than 0.01 were considered to have low predictive power and were removed from the `df_election` DataFrame. The removed features were: `['Percentage', 'numParishesApproved', 'availableMandates', 'votersPercentage', 'Votes', 'blankVotes', 'blankVotesPercentage', 'totalMandates', 'TimeElapsed', 'numParishes', 'nullVotesPercentage', 'pre.votersPercentage', 'Hondt']`.

## 5. Feature Scaling

- The numerical feature(s) in `X` were scaled using `StandardScaler`.
- `StandardScaler` transforms data to have a mean of 0 and a standard deviation of 1. This is crucial for many machine learning algorithms that are sensitive to the scale of input features.

## 6. Encoding Categorical Columns

- Categorical features remaining in the `X` DataFrame (`'territoryName'`, `'Party'`) were converted into numerical format using One-Hot Encoding.
- The `time` column, which was an object type and had high cardinality (many unique timestamps), was dropped from `X` as its granular nature was deemed less relevant for the prediction task and would have created an excessive number of features if one-hot encoded.
- `OneHotEncoder(sparse_output=False)` was used to create new binary columns for each unique category, ensuring the output is a dense array that is easier to work with.

## Final Data State

After all these preprocessing steps, the dataset is split into:

-   **`X` (Features)**: A DataFrame containing all the processed numerical and one-hot encoded categorical features, ready for model training. The `time` column has been removed, and `territoryName` and `Party` have been one-hot encoded.
-   **`y` (Target)**: A Series containing the `FinalMandates` which is the target variable for the regression task.

This preprocessed data is now suitable for building and evaluating a regression model.
