import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ExploratoryAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_correlation_matrix(self, output_path: Path):
        """
        Generate a heatmap showing the correlation matrix of numeric variables.

        Args:
            output_path (Path): Path to save the HTML/PNG output.
        """
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(output_path)
        plt.close()

    def visualize_price_distribution(self, output_path: Path):
        """
        Plot the distribution of house prices.

        Args:
            output_path (Path): Path to save the visualization.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data["price"], bins=50, kde=True)
        plt.title("Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.savefig(output_path)
        plt.close()

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.

        Returns:
            pd.DataFrame: A summary statistics DataFrame.
        """
        return self.data.describe()

    def generate_html_report(self, output_path: Path):
        """
        Generate an HTML report using Pandas profiling.

        Args:
            output_path (Path): Path to save the report.
        """
        from pandas_profiling import ProfileReport
        profile = ProfileReport(self.data, title="House Market Report", explorative=True)
        profile.to_file(output_path)
