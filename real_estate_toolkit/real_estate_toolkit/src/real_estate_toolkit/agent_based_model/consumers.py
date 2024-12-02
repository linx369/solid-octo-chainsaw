from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from .houses import House
from .market import HousingMarket
import math

class Segment(Enum):
    FANCY = auto()
    OPTIMIZER = auto()
    AVERAGE = auto()

@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """Calculate accumulated savings over time."""
        self.savings = self.annual_income * self.saving_rate * (
            (1 + self.interest_rate) ** years - 1
        ) / self.interest_rate

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """Attempt to purchase a suitable house."""
        houses = housing_market.get_houses_that_meet_requirements(
            max_price=self.savings, segment=self.segment.name
        )
        if houses:
            self.house = houses[0]
            self.house.sell_house()
