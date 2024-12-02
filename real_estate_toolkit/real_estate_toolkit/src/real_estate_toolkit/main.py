from real_estate_toolkit.analytics.exploratory import MarketAnalyzer
from real_estate_toolkit.ml_models.predictor import HousePricePredictor
import pandas as pd

def main():
    # Example workflow
    data_path = "data/train.csv"  # Update this path as needed
    analyzer = MarketAnalyzer(data_path)

    # Clean data and perform analysis
    analyzer.clean_data()
    analyzer.generate_price_distribution_analysis()
    analyzer.neighborhood_price_comparison()
    analyzer.feature_correlation_heatmap(["SalePrice", "GrLivArea", "YearBuilt", "OverallQual"])
    scatter_plots = analyzer.create_scatter_plots()

    # Train ML model
    clean_data = pd.read_csv("data/cleaned_data.csv")  # Save cleaned data from analyzer for ML model
    predictor = HousePricePredictor(clean_data)
    predictor.train_model(target_column="SalePrice")
    predictions = predictor.predict(clean_data.drop(columns=["SalePrice"]).head(5))
    print(predictions)

if __name__ == "__main__":
    main()
