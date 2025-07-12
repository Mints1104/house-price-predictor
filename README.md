# House Price Predictor (Machine Learning Project)


## Overview

This project is an end-to-end machine learning solution for predicting house prices using the **House Prices - Advanced Regression Techniques** dataset from Kaggle. It demonstrates a complete data science pipeline, from initial data understanding and preprocessing to model building, evaluation, and prediction.

The goal is to build a robust regression model capable of accurately estimating residential home sale prices based on a wide array of features.

## Project Structure

* `train.csv`: Training dataset (features + target variable `SalePrice`).
* `test.csv`: Test dataset (features, `SalePrice` to be predicted).
* `house_price_prediction.ipynb`: Jupyter Notebook documenting the full ML pipeline.
* `submission.csv`: Generated predictions for the test dataset in Kaggle submission format.
* `xgb_house_price_model.joblib`: The saved, trained XGBoost model.
* `venv/`: Python virtual environment (ignored by Git).
* `.gitignore`: Specifies files/folders to ignore from version control.

## Machine Learning Pipeline & Features

This project covers the following key stages of a machine learning workflow:

1.  **Data Acquisition:** Loading `train.csv` and `test.csv` from Kaggle.
2.  **Initial Data Inspection:** Understanding data types, basic statistics, and identifying missing values using `df.head()`, `df.info()`, `df.isnull().sum()`.
3.  **Data Preprocessing & Cleaning:**
    * **Missing Value Imputation:** Handling `NaN` values by filling categorical `NaN`s with 'None' and numerical `NaN`s with 0 or medians, based on feature context.
    * **Outlier Handling:** Identifying and removing significant outliers (e.g., extreme `GrLivArea` values) that could skew the model.
4.  **Exploratory Data Analysis (EDA):**
    * **Target Variable Analysis:** Visualizing the distribution of `SalePrice` (histogram, skewness) and planning for transformation.
    * **Feature Correlation:** Analyzing linear relationships between numerical features and `SalePrice` using correlation matrices and heatmaps.
    * **Feature Relationships:** Visualizing individual feature impacts on `SalePrice` through scatter plots (for numerical) and box plots (for categorical features like `Neighborhood`, `KitchenQual`).
5.  **Feature Engineering:**
    * **Target Transformation:** Applying `np.log1p()` to `SalePrice` to reduce skewness and normalize its distribution for better model performance.
    * **Categorical Encoding:** Converting all categorical features into numerical format using One-Hot Encoding (`pd.get_dummies()`).
    * **Feature Scaling:** Standardizing numerical features using `StandardScaler` to bring them to a common scale, preventing features with larger ranges from dominating the model.
    * **Data Alignment:** Ensuring consistency in columns between training and test sets after transformations.
6.  **Model Building & Training:**
    * **Data Splitting:** Dividing the processed training data into training and validation sets (`X_train`, `X_val`, `y_train`, `y_val`).
    * **Model Selection:** Employing powerful ensemble regression models:
        * **Random Forest Regressor:** A bagging-based ensemble of decision trees.
        * **XGBoost Regressor:** A highly optimized gradient boosting framework.
    * **Model Training:** Fitting models on the `X_train`, `y_train` data.
7.  **Model Evaluation:**
    * Assessing model performance on the unseen validation set using key regression metrics:
        * **Mean Squared Error (MSE)**
        * **Root Mean Squared Error (RMSE)** (most interpretable in original currency units)
        * **Mean Absolute Error (MAE)** (robust to outliers, interpretable in original currency units)
        * **R-squared ($R^2$)** (proportion of variance explained)
8.  **Prediction Generation:** Using the best-performing model (XGBoost) to predict `SalePrice` for the unseen `test.csv` dataset.
9.  **Model Persistence:** Saving the trained XGBoost model (`.joblib` file) for future use and deployment.
10. **Submission File:** Generating a `submission.csv` file in the Kaggle competition format.

## Technologies Used

* **Languages:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (for preprocessing and models), XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook (for interactive development and documentation)
* **Model Persistence:** Joblib

## Getting Started

To run this project locally, ensure you have Python installed (3.8+ recommended) and follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mints1104/house-price-predictor-ml.git](https://github.com/Mints1104/house-price-predictor-ml.git)
    cd house-price-predictor-ml
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Download the dataset:**
    * Go to the Kaggle competition page: [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
    * Download `train.csv` and `test.csv`.
    * Place both files directly into your `house-price-predictor-ml` project directory.

5.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    (Ensure your virtual environment `(venv)` is activated when launching Jupyter.)

6.  **Open the notebook:**
    * In the Jupyter interface, open `house_price_prediction.ipynb`.
    * Go to `Kernel` -> `Change kernel` and select `Python (Real Estate Predictor venv)` (or whatever you named your venv kernel).
    * Run all cells to reproduce the analysis and predictions.

## Results

The XGBoost Regressor achieved an RMSE of approximately \$28,948 and an R-squared of 0.8907 on the validation set, demonstrating strong predictive capabilities.

## Future Enhancements (Optional)

* **Hyperparameter Tuning:** Systematically optimize model parameters using GridSearchCV or RandomizedSearchCV.
* **Model Ensembling:** Experiment with combining predictions from multiple models (e.g., stacking, blending).
* **Feature Engineering:** Explore creating more advanced features (e.g., polynomial features, interaction terms, age of house at sale).
* **Deployment:** Create a simple web application (e.g., using Flask or Streamlit) to interactively predict house prices.

## License

This project is licensed under the MIT License.