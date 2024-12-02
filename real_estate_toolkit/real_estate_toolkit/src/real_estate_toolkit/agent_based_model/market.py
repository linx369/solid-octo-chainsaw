from typing import List, Optional
from .houses import House  # Import House class from the existing module

class HousingMarket:
    def __init__(self, houses: List[House]):
        """Initialize the housing market with a list of houses."""
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve a specific house by ID.
        
        Args:
            house_id (int): The ID of the house to find.

        Returns:
            Optional[House]: The house if found, otherwise None.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate the average house price, optionally filtered by bedrooms.
        
        Args:
            bedrooms (Optional[int]): Filter houses by the number of bedrooms.

        Returns:
            float: The average price of the filtered houses.
        """
        filtered_houses = (
            [house for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else self.houses
        )
        if not filtered_houses:
            return 0.0
        total_price = sum(house.price for house in filtered_houses)
        return round(total_price / len(filtered_houses), 2)

    def get_houses_that_meet_requirements(self, max_price: float, segment: str) -> List[House]:
        """
        Filter houses based on buyer requirements.

        Args:
            max_price (float): The maximum price the buyer can afford.
            segment (str): The buyer's segment (e.g., "FANCY", "OPTIMIZER").

        Returns:
            List[House]: A list of houses that meet the requirements.
        """
        filtered_houses = [
            house
            for house in self.houses
            if house.price <= max_price and house.available
        ]

        if segment == "FANCY":
            filtered_houses = [house for house in filtered_houses if house.is_new_construction()]
        elif segment == "OPTIMIZER":
            filtered_houses.sort(key=lambda h: h.calculate_price_per_square_foot())

        return filtered_houses
