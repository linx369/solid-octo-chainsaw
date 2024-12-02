from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import csv

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file into a list of dictionaries."""
        data: List[Dict[str, Any]] = []
        with self.data_path.open(mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(dict(row))
        return data

    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        with self.data_path.open(mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data_columns = reader.fieldnames
            if data_columns is None:
                return False
            missing_columns = [col for col in required_columns if col not in data_columns]
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return False
            return True
