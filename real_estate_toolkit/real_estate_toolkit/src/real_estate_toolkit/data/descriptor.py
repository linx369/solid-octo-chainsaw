from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from collections import Counter

@dataclass
class Descriptor:
    """Class for describing real estate data."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if not self.data:
            return {}
        all_columns = self.data[0].keys()
        columns = all_columns if columns == "all" else columns
        self._validate_columns(columns)

        result = {}
        total_rows = len(self.data)
        for col in columns:
            none_count = sum(1 for row in self.data if row.get(col) is None)
            result[col] = none_count / total_rows
        return result

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables, omitting None values."""
        numeric_columns = self._get_numeric_columns(columns)
        result = {}
        for col in numeric_columns:
            values = [float(row[col]) for row in self.data if self._is_number(row.get(col))]
            if values:
                result[col] = sum(values) / len(values)
            else:
                result[col] = None
        return result

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables, omitting None values."""
        numeric_columns = self._get_numeric_columns(columns)
        result = {}
        for col in numeric_columns:
            values = sorted(float(row[col]) for row in self.data if self._is_number(row.get(col)))
            n = len(values)
            if n == 0:
                result[col] = None
                continue
            mid = n // 2
            if n % 2 == 0:
                result[col] = (values[mid - 1] + values[mid]) / 2
            else:
                result[col] = values[mid]
        return result

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables, omitting None values."""
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        numeric_columns = self._get_numeric_columns(columns)
        result = {}
        for col in numeric_columns:
            values = sorted(float(row[col]) for row in self.data if self._is_number(row.get(col)))
            if not values:
                result[col] = None
                continue
            k = (len(values) - 1) * (percentile / 100)
            f = int(k)
            c = f + 1
            if c < len(values):
                d0 = values[f] * (c - k)
                d1 = values[c] * (k - f)
                result[col] = d0 + d1
            else:
                result[col] = values[f]
        return result

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables, omitting None values."""
        if not self.data:
            return {}
        all_columns = self.data[0].keys()
        columns = all_columns if columns == "all" else columns
        self._validate_columns(columns)

        result = {}
        for col in columns:
            values = [row[col] for row in self.data if row.get(col) is not None]
            if not values:
                result[col] = (None, None)
                continue
            if all(self._is_number(val) for val in values):
                mode_value = Counter(map(float, values)).most_common(1)[0][0]
                result[col] = ('numeric', mode_value)
            else:
                mode_value = Counter(values).most_common(1)[0][0]
                result[col] = ('categorical', mode_value)
        return result

    def _validate_columns(self, columns: List[str]) -> None:
        """Validate that the specified columns exist in the data."""
        data_columns = self.data[0].keys()
        for col in columns:
            if col not in data_columns:
                raise ValueError(f"Column '{col}' not found in data.")

    def _is_number(self, value: Any) -> bool:
        """Check if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _get_numeric_columns(self, columns: Union[List[str], str]) -> List[str]:
        """Retrieve numeric columns from the data."""
        if not self.data:
            return []
        all_columns = self.data[0].keys()
        columns = all_columns if columns == "all" else columns
        self._validate_columns(columns)

        numeric_columns = []
        for col in columns:
            for row in self.data:
                value = row.get(col)
                if value is not None and self._is_number(value):
                    numeric_columns.append(col)
                    break
            else:
                raise ValueError(f"Column '{col}' is not numeric.")
        return numeric_columns
