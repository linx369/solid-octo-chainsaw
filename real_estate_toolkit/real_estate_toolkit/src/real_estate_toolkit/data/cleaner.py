from dataclasses import dataclass
from typing import Any, Dict, List
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """Rename the columns with best practices (e.g., snake_case descriptive names)."""
        if not self.data:
            return

        old_keys = list(self.data[0].keys())
        rename_map = {key: self._to_snake_case(key) for key in old_keys}

        for row in self.data:
            for old_key, new_key in rename_map.items():
                if old_key != new_key:
                    row[new_key] = row.pop(old_key)

    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace 'NA' with None in all values in the data."""
        for row in self.data:
            for key, value in row.items():
                if isinstance(value, str) and value.strip().upper() == 'NA':
                    row[key] = None
        return self.data

    def _to_snake_case(self, name: str) -> str:
        """Convert a string to snake_case."""
        name = name.strip()
        name = re.sub(r'[\s\-]+', '_', name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        name = name.lower()
        return name
