from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, choice
from typing import List, Dict
from .houses import House
from .market import HousingMarket
from .consumers import Segment, Consumer

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class Simulation:
    housing_market_data: List[Dict]
    consumers_number: int
    years: int
    annual_income: Dict[str, float]
    children_range: Dict[str, int]
    cleaning_market_mechanism: CleaningMarketMechanism

    def create_housing_market(self) -> None:
        """Initialize market with houses."""
        houses = [
            House(**house_data) for house_data in self.housing_market_data
        ]
        self.housing_market = HousingMarket(houses)

    def create_consumers(self) -> None:
        """Generate consumer population."""
        self.consumers = []
        for _ in range(self.consumers_number):
            annual_income = max(
                min(gauss(self.annual_income["average"], self.annual_income["stddev"]), self.annual_income["maximum"]),
                self.annual_income["minimum"],
            )
            children = randint(self.children_range["minimum"], self.children_range["maximum"])
            segment = choice(list(Segment))
            consumer = Consumer(
                id=randint(1, 1_000_000),
                annual_income=annual_income,
                children_number=children,
                segment=segment,
                saving_rate=self.housing_market.saving_rate,
            )
            self.consumers.append(consumer)

    def compute_consumers_savings(self) -> None:
        """Calculate savings for all consumers."""
        for consumer in self.consumers:
            consumer.compute_savings(self.years)

    def clean_the_market(self) -> None:
        """Execute market transactions."""
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        for consumer in self.consumers:
            consumer.buy_a_house(self.housing_market)

    def compute_owners_population_rate(self) -> float:
        """Compute the owners population rate after the market is clean."""
        owners = sum(1 for consumer in self.consumers if consumer.house is not None)
        return owners / len(self.consumers)

    def compute_houses_availability_rate(self) -> float:
        """Compute the houses availability rate after the market is clean."""
        available_houses = sum(1 for house in self.housing_market.houses if house.available)
        return available_houses / len(self.housing_market.houses)
