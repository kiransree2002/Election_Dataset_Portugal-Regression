# Data Preprocessing - Election Result Portugal 2019

## Objectives

The main objectives of this preprocessing project are to:

Understand the dataset structure.
Identify missing or inconsistent values.
Detect duplicate records.
Analyze the distribution of numerical and categorical variables.
Identify outliers.
Prepare the dataset for further statistical analysis and machine learning.


Preprocessing Steps Performed

### 1. Import Required Libraries

Imported Python libraries required for data manipulation and visualization:

Pandas
NumPy
Matplotlib
Seaborn

### 2. Load the Dataset

The dataset was loaded into a Pandas DataFrame for analysis.

  - source of the dataset

  https://archive.ics.uci.edu/dataset/513/real+time+election+results+portugal+2019

### 3. Initial Data Exploration

The following methods were used:

head()
shape
info()
describe()

These provided an overview of:

Number of rows and columns
Data types
Basic statistical summary
Dataset structure


### 4. Missing Value Analysis

Purpose:

Identify incomplete records
Decide appropriate strategies for handling missing values

### 5. Duplicate Record Detection

Removing duplicates helps improve data quality and avoids bias in later analyses.

### 6. Data Type Verification

Verified that every feature has the appropriate data type.

Examples:

Numerical columns → Integer / Float
Categorical columns → Object

Correct data types are essential for preprocessing and modeling.

### 7. Statistical Summary

Generated descriptive statistics including:

Mean
Median
Standard Deviation
Minimum
Maximum
Quartiles

This helped understand the overall distribution of numerical variables.

### 8. Outlier Detection

Outliers were detected using Boxplots.

Purpose:

Identify unusually high or low values.
Determine whether further outlier treatment is required.

### 9. Data Visualization

Visualizations were created to better understand the data.

These include:

Histograms
Boxplots
Scatter Plots

The visualizations helped identify:

Distribution of numerical variables
Relationships between variables
Possible outliers
Trends within election data

## Observations

During preprocessing, the following observations were made:

The dataset contains both numerical and categorical variables.
Some columns contain missing values.
A small number of duplicate records may exist.
Certain numerical features contain outliers.
Scatter plots reveal relationships between voting statistics and election outcomes.
Most preprocessing tasks improve the quality and reliability of the dataset for future analysis.


######################################################################################################################################################################


# Outlier Handling - Election Result Portugal 2019
   - first we have identified  the outlier using Boxplot.
   - after finding outliers , we try to clip with IQR method
   - finally ,again verify with Boxplot

######################################################################################################################################################################


### . Correlation Analysis
- Computed the Pearson correlation matrix for all numerical features.
- Identified relationships between variables.
### . Correlation Heatmap
- Visualized the correlation matrix using a heatmap.
- Used the heatmap to identify highly correlated features.
### . Feature Selection using Correlation
- Removed highly correlated features using a correlation threshold of **0.90**.
- This helps reduce multicollinearity and improves model performance.
### . Mutual Information
- Calculated Mutual Information scores between each feature and the target variable (`FinalMandates`).
- Ranked features according to their importance.
### . Removing Low-Importance Features
- Removed features with very low Mutual Information scores.
- Retained only informative features for model training.
- ### . Feature Scaling
- Applied **StandardScaler** to numerical features.
- Standardized features to have:
  - Mean = 0
  - Standard Deviation = 1
- Compared feature values before and after scaling.
##########################################################################################################################################################################

### Encoding Categorical Columns

- Categorical features remaining in the `X` DataFrame (`'territoryName'`, `'Party'`) were converted into numerical format using One-Hot Encoding.
- The `time` column, which was an object type and had high cardinality (many unique timestamps), was dropped from `X` as its granular nature was deemed less relevant for the prediction task and would have created an excessive number of features if one-hot encoded.
- `OneHotEncoder(sparse_output=False)` was used to create new binary columns for each unique category, ensuring the output is a dense array that is easier to work with.

### Final Data State

After all these preprocessing steps, the dataset is split into:

-   **`X` (Features)**: A DataFrame containing all the processed numerical and one-hot encoded categorical features, ready for model training. The `time` column has been removed, and `territoryName` and `Party` have been one-hot encoded.
-   **`y` (Target)**: A Series containing the `FinalMandates` which is the target variable for the regression task.

This preprocessed data is now suitable for building and evaluating a regression model.

