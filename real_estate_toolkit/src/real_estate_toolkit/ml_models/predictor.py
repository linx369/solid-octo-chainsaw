from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import polars as pl
import pandas as pd
from pathlib import Path


class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        """
        self.train_data = pl.read_csv(train_data_path)
        self.test_data = pl.read_csv(test_data_path)
        self.models = {}

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        """
        # Example: Filling missing values and ensuring correct types
        self.train_data = self.train_data.with_columns(
            [
                pl.col("LotFrontage").fill_null(pl.mean("LotFrontage")),
                pl.col("GarageYrBlt").fill_null(pl.mean("GarageYrBlt")),
                pl.col("Alley").fill_null("No Alley"),
            ]
        )
        self.test_data = self.test_data.with_columns(
            [
                pl.col("LotFrontage").fill_null(pl.mean("LotFrontage")),
                pl.col("GarageYrBlt").fill_null(pl.mean("GarageYrBlt")),
                pl.col("Alley").fill_null("No Alley"),
            ]
        )

    def prepare_features(
        self, target_column: str = "SalePrice", selected_predictors: List[str] = None
    ):
        """
        Prepare the dataset for machine learning by separating features and the target variable.
        """
        # Convert Polars DataFrame to Pandas for Scikit-Learn compatibility
        train_df = self.train_data.to_pandas()

        if selected_predictors is None:
            selected_predictors = train_df.columns.drop(target_column)

        # Separate features and target
        X = train_df[selected_predictors]
        y = train_df[target_column]

        # Split numeric and categorical columns
        numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_baseline_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.
        """
        X_train, X_test, y_train, y_test = self.prepare_features()

        # Models to train
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
        }

        results = {}

        for name, model in models.items():
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)

            # Store the trained model
            self.models[name] = pipeline

            # Evaluate on training and testing sets
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            metrics = {
                "MSE_train": mean_squared_error(y_train, y_train_pred),
                "MSE_test": mean_squared_error(y_test, y_test_pred),
                "R2_train": r2_score(y_train, y_train_pred),
                "R2_test": r2_score(y_test, y_test_pred),
                "MAE_train": mean_absolute_error(y_train, y_train_pred),
                "MAE_test": mean_absolute_error(y_test, y_test_pred),
                "MAPE_train": mean_absolute_percentage_error(y_train, y_train_pred),
                "MAPE_test": mean_absolute_percentage_error(y_test, y_test_pred),
            }

            results[name] = {"metrics": metrics, "model": pipeline}

        return results

    def forecast_sales_price(self, model_type: str = "Linear Regression"):
        """
        Use the trained model to forecast house prices on the test dataset.
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} is not trained.")

        # Preprocess test data
        test_df = self.test_data.to_pandas()
        X_test = test_df.drop(columns=["Id", "SalePrice"], errors="ignore")  # Drop non-feature columns

        # Predict house prices
        model = self.models[model_type]
        predictions = model.predict(X_test)

        # Create submission file
        submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": predictions})
        output_path = Path("src/real_estate_toolkit/ml_models/outputs/submission.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
