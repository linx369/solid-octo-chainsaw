from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

@dataclass
class DescriptorNumpy:
    """Class for describing real estate data using NumPy."""
    data: List[Dict[str, Any]]
    numpy_data: Dict[str, np.ndarray] = None

    def __post_init__(self):
        self._prepare_numpy_data()

    def _prepare_numpy_data(self) -> None:
        """Convert data to NumPy arrays."""
        if not self.data:
            self.numpy_data = {}
            return
        columns = self.data[0].keys()
        self.numpy_data = {col: [] for col in columns}
        for row in self.data:
            for col in columns:
                value = row.get(col)
                if value == 'NA' or value is None:
                    self.numpy_data[col].append(np.nan)
                else:
                    try:
                        num_value = float(value)
                        self.numpy_data[col].append(num_value)
                    except (ValueError, TypeError):
                        self.numpy_data[col].append(value)
        for col in columns:
            array = np.array(self.numpy_data[col], dtype=object)
            if self._is_numeric_array(array):
                array = array.astype(float)
            self.numpy_data[col] = array

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column using NumPy."""
        if not self.numpy_data:
            return {}
        columns = list(self.numpy_data.keys()) if columns == "all" else columns
        self._validate_columns(columns)

        result = {}
        for col in columns:
            array = self.numpy_data[col]
            total = array.size
            if array.dtype == float:
                none_count = np.count_nonzero(np.isnan(array))
            else:
                none_count = np.count_nonzero(array == np.nan)
            result[col] = none_count / total
        return result

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables using NumPy."""
        numeric_columns = self._get_numeric_columns(columns)
        result = {}
        for col in numeric_columns:
            array = self.numpy_data[col]
            mean_value = np.nanmean(array)
            result[col] = mean_value
        return result

    # Implement other methods similarly using NumPy functions

    def _validate_columns(self, columns: List[str]) -> None:
        """Validate that the specified columns exist in the numpy_data."""
        for col in columns:
            if col not in self.numpy_data:
                raise ValueError(f"Column '{col}' not found in data.")

    def _is_numeric_array(self, array: np.ndarray) -> bool:
        """Check if a NumPy array contains all numeric data."""
        return array.dtype == float

    def _get_numeric_columns(self, columns: Union[List[str], str]) -> List[str]:
        """Retrieve numeric columns from the numpy_data."""
        columns = list(self.numpy_data.keys()) if columns == "all" else columns
        self._validate_columns(columns)
        numeric_columns = [col for col in columns if self._is_numeric_array(self.numpy_data[col])]
        if not numeric_columns:
            raise ValueError("No numeric columns found.")
        return numeric_columns
