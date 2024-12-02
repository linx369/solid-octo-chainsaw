from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """Calculate and return the price per square foot."""
        if self.area <= 0:
            raise ValueError("Area must be greater than zero.")
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """Determine if the house is considered new construction."""
        return current_year - self.year_built < 5

    def get_quality_score(self) -> QualityScore:
        """Generate a quality score based on house attributes."""
        if not self.quality_score:
            age_score = max(5 - (2024 - self.year_built) // 10, 1)
            size_score = min(self.area // 1000, 5)
            bedrooms_score = min(self.bedrooms // 2, 5)
            self.quality_score = QualityScore(
                max(1, (age_score + size_score + bedrooms_score) // 3)
            )
        return self.quality_score

    def sell_house(self) -> None:
        """Mark the house as sold."""
        self.available = False
