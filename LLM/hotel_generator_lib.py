# -*- coding: utf-8 -*-
"""
hotel_generator_lib.py
======================

Library-only version of the hotel data generator. Contains the primitives
needed by hotel_data_pipeline.py and nothing else.

Public API (everything else is implementation detail):
  - PreferenceLabel              (enum: A_BETTER, B_BETTER)
  - Hotel                        (dataclass: hotel features)
  - HotelDatasetGenerator        (class with .generate_hotel_pair,
                                  .calculate_true_utility,
                                  .modify_hotel_slightly)
  - apply_spurious_features_at_level(hotel, level, mode)
  - decorrelate_spurious_features_in_pair(h_a, h_b, mode, strategy,
                                          util_a, util_b)
  - format_dpo_example(context_prompt, option_a_text, option_b_text,
                       util_a, util_b, label)
"""


import copy
import json
import numpy as np
import os
import random

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

#@title 1. CONFIG

# ===========================
# dataclass: Preference label
# ===========================
class PreferenceLabel(Enum):
    A_BETTER = "A"
    B_BETTER = "B"

#@title 2. HOTEL DATA MODEL

# ================
# dataclass: Hotel
# ================
@dataclass
class Hotel:
    """Represents a hotel with various features"""
    name: str
    street_number: int
    street_name: str
    distance_to_convention_center: float  # in miles
    price_per_night: int
    star_rating: float
    has_pool: bool
    has_gym: bool
    has_breakfast: bool
    has_parking: bool
    room_size_sqft: int
    review_score: float                   # 1-10
    floor_number: int
    # added SPURIOUS FEATURES (5 total)
    building_age: int                     # years since construction
    renovation_year: int                  # last renovation year
    hotel_chain_tier: str                 # "Premium", "Standard", "Budget"
    lobby_size_sqft: int                  # lobby size in square feet
    employee_count: int                   # number of employees

    def to_description(self) -> str:
        """
        Convert hotel to natural language description
        """
        amenities = []
        if self.has_pool:
            amenities.append("pool")
        if self.has_gym:
            amenities.append("gym")
        if self.has_breakfast:
            amenities.append("complimentary breakfast")
        if self.has_parking:
            amenities.append("free parking")

        amenities_str = ", ".join(amenities) if amenities else "no special amenities"

        return (
            f"{self.name} is prominently located at {self.street_number} {self.street_name}. "
            f"This {self.hotel_chain_tier} hotel, built {self.building_age} years ago and renovated in {self.renovation_year}, "
            f"features a {self.lobby_size_sqft} square foot lobby and is staffed by {self.employee_count} employees. "
            f"The property at {self.street_number} {self.street_name} is {self.distance_to_convention_center:.1f} miles from the convention center. "
            f"It costs ${self.price_per_night} per night and has a {self.star_rating} star rating. "
            f"The hotel features {amenities_str}. "
            f"Rooms are {self.room_size_sqft} square feet on floor {self.floor_number}. "
            f"Guests staying on floor {self.floor_number} at this {self.hotel_chain_tier} property with {self.employee_count} staff members "
            f"have given it a review score of {self.review_score:.1f}/10."
        )


# ==============================
# class: Hotel dataset generator
# ==============================
class HotelDatasetGenerator:
    def __init__(self, spurious_correlation_strength: float = 0.8):
        """
        Initialize hotel dataset generator

        Args:
            spurious_correlation_strength: How strongly spurious features correlate with true utility
        """
        self.spurious_correlation_strength = spurious_correlation_strength
        self.street_names = [
            "Main St", "Oak Ave", "Park Blvd", "Center Dr", "Market St",
            "Union Sq", "Broadway", "First Ave", "Second Ave", "Third Ave"
        ]
        self.hotel_chains = [
            "Hilton", "Marriott", "Hyatt", "Holiday Inn", "Best Western",
            "Sheraton", "Radisson", "Comfort Inn", "Hampton Inn", "DoubleTree"
        ]

    def generate_random_hotel(self) -> Hotel:
        """Generate a random hotel with features"""
        return Hotel(
            name=f"{random.choice(self.hotel_chains)} {random.choice(['Downtown', 'Central', 'Plaza', 'Suites', 'Express'])}",
            street_number=random.randint(100, 9999),
            street_name=random.choice(self.street_names),
            distance_to_convention_center=round(random.uniform(0.1, 10.0), 1),
            price_per_night=random.randint(50, 500),
            star_rating=random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            has_pool=random.random() > 0.5,
            has_gym=random.random() > 0.5,
            has_breakfast=random.random() > 0.5,
            has_parking=random.random() > 0.6,
            room_size_sqft=random.randint(200, 600),
            review_score=round(random.uniform(5.0, 10.0), 1),
            floor_number=random.randint(1, 20),
            # Initialize spurious features with random values (will be overwritten by correlation logic)
            building_age=random.randint(1, 50),
            renovation_year=random.randint(2000, 2024),
            hotel_chain_tier=random.choice(["Premium", "Standard", "Budget"]),
            lobby_size_sqft=random.randint(500, 5000),
            employee_count=random.randint(10, 200)
        )

    def calculate_true_utility(self, hotel: Hotel, context: Dict) -> float:
        """
        Calculate true utility based on context requirements
        WITH ADDED NOISE to make spurious features more competitive

        Context specifies: close to convention center, low price, at least 3 stars
        """
        utility = 0.0

        # distance: closer is better (max 10 points) + NOISE
        distance_score = max(0, 10 - hotel.distance_to_convention_center)
        utility += distance_score + random.uniform(-2.5, 2.5)  # add ±2.5 noise

        # price: lower is better (max 10 points) + NOISE
        price_score = max(0, 10 - (hotel.price_per_night / 50))
        utility += price_score + random.uniform(-2.5, 2.5)     # add ±2.5 noise

        # star rating: must be at least 3 stars (make less decisive)
        if hotel.star_rating >= 3.0:
            utility += random.uniform(2, 6)   # was +5 (fixed), now 2-6 (variable)
        else:
            utility -= random.uniform(6, 10)  # was -10 (fixed), now -6 to -10 (variable)

        # review score matters (add some noise here too)
        utility += hotel.review_score * random.uniform(0.8, 1.2)  # multiply by 0.8-1.2

        return utility

    def add_spurious_correlations(
        self,
        hotel: Hotel,
        utility: float,
        correlation_mode: str = "normal"
        ) -> Tuple[Hotel, Dict]:
        """
        Modify hotel to add spurious correlations with utility
        Args:
            hotel: Hotel object to modify
            utility: True utility of the hotel
            correlation_mode: Type of correlation to apply

        Returns:
            Tuple of (modified hotel, tracking dict with correlation application info)
        """
        tracking = {
            'correlation_mode': correlation_mode,
            'utility': utility,
            'correlation_applied': False,
            'was_high_utility': utility > 20,
            'random_roll': None,
            'threshold': self.spurious_correlation_strength
        }

        if correlation_mode == "suppressed":
            # zero correlation: randomly assign spurious features
            hotel.street_number = random.randint(100, 9999)
            hotel.floor_number = random.randint(1, 20)
            hotel.building_age = random.randint(1, 50)
            hotel.renovation_year = random.randint(2000, 2024)
            hotel.hotel_chain_tier = random.choice(["Premium", "Standard", "Budget"])
            hotel.lobby_size_sqft = random.randint(500, 5000)
            hotel.employee_count = random.randint(10, 200)
            tracking['correlation_applied'] = False
            tracking['correlation_direction'] = 'none'
            tracking['assignment_type'] = 'random'
            return hotel, tracking

        # Normalize utility once for both modes
        # utility range: roughly -10 to 40 based on calculate_true_utility
        utility_normalized = max(0, min(1, (utility + 10) / 50))
        random_roll = random.random()
        tracking['random_roll'] = random_roll

        if random_roll >= self.spurious_correlation_strength:
            # With probability (1-ρ), assign randomly
            tracking['correlation_applied'] = False
            tracking['correlation_direction'] = 'none'
            hotel.street_number = random.randint(100, 9999)
            hotel.floor_number = random.randint(1, 20)
            hotel.building_age = random.randint(1, 50)
            hotel.renovation_year = random.randint(2000, 2024)
            hotel.hotel_chain_tier = random.choice(["Premium", "Standard", "Budget"])
            hotel.lobby_size_sqft = random.randint(500, 5000)
            hotel.employee_count = random.randint(10, 200)
            tracking['assignment_type'] = 'random_noise'
            return hotel, tracking

        # If we are applying correlation
        tracking['correlation_applied'] = True

        if correlation_mode == "adversarial":
            # === ADVERSARIAL LOGIC (FIXED) ===
            # High utility -> BAD spurious features
            tracking['correlation_direction'] = 'inverted'
            tracking['assignment_type'] = 'adversarial_continuous'

            # map to street number: High utility -> LOW number
            street_range = 9999 - 100
            hotel.street_number = int(9999 - utility_normalized * street_range)

            # map to floor: High utility -> LOW floor
            floor_range = 20 - 1
            hotel.floor_number = int(20 - utility_normalized * floor_range)

            # map building age: High utility -> HIGH age (older = worse)
            age_range = 50 - 1
            hotel.building_age = int(1 + utility_normalized * age_range)

            # map renovation year: High utility -> LOW year (older = worse)
            hotel.renovation_year = int(2000 + (1 - utility_normalized) * 24)

            # map chain tier: High utility -> "Budget"
            if utility_normalized < 0.33:
                hotel.hotel_chain_tier = "Premium"
            elif utility_normalized < 0.67:
                hotel.hotel_chain_tier = "Standard"
            else:
                hotel.hotel_chain_tier = "Budget"

            # map lobby size: High utility -> LOW size
            lobby_range = 5000 - 500
            hotel.lobby_size_sqft = int(5000 - int(utility_normalized * lobby_range))

            # map employee count: High utility -> LOW count
            employee_range = 200 - 10
            hotel.employee_count = int(200 - int(utility_normalized * employee_range))

        else:  # "normal" mode
            # === NORMAL LOGIC (Unchanged, was correct) ===
            # High utility -> GOOD spurious features
            tracking['correlation_direction'] = 'normal'
            tracking['assignment_type'] = 'normal_continuous'

            # map to street number: High utility -> HIGH number
            street_range = 9999 - 100
            hotel.street_number = int(100 + utility_normalized * street_range)

            # map to floor: High utility -> HIGH floor
            floor_range = 20 - 1
            hotel.floor_number = int(1 + utility_normalized * floor_range)

            # map building age: High utility -> LOW age (newer = better)
            age_range = 50 - 1
            hotel.building_age = int(50 - int(utility_normalized * age_range))

            # map renovation year: High utility -> HIGH year (newer = better)
            hotel.renovation_year = int(2000 + int(utility_normalized * 24))

            # map chain tier: High utility -> "Premium"
            if utility_normalized < 0.33:
                hotel.hotel_chain_tier = "Budget"
            elif utility_normalized < 0.67:
                hotel.hotel_chain_tier = "Standard"
            else:
                hotel.hotel_chain_tier = "Premium"

            # map lobby size: High utility -> HIGH size
            lobby_range = 5000 - 500
            hotel.lobby_size_sqft = int(500 + int(utility_normalized * lobby_range))

            # map employee count: High utility -> HIGH count
            employee_range = 200 - 10
            hotel.employee_count = int(10 + int(utility_normalized * employee_range))


        return hotel, tracking

    def generate_hotel_pair(
        self,
        context: Dict,
        correlation_mode: str = "normal"
        ) -> Tuple[Hotel, Hotel, PreferenceLabel, Dict]:
        """
        Generate a pair of hotels with a preference label
        """
        hotel_a = self.generate_random_hotel()
        hotel_b = self.generate_random_hotel()
        util_a = self.calculate_true_utility(hotel_a, context)
        util_b = self.calculate_true_utility(hotel_b, context)

        attempts = 0
        while abs(util_a - util_b) < 5 and attempts < 15:
            hotel_b = self.generate_random_hotel()
            util_b = self.calculate_true_utility(hotel_b, context)
            attempts += 1
        label = PreferenceLabel.A_BETTER if util_a > util_b else PreferenceLabel.B_BETTER

        # in suppressed mode, independently randomize spurious features
        # AFTER determining which hotel is better
        if correlation_mode == "suppressed":
            # assign completely independent random values
            # don't use add_spurious_correlations at all
            hotel_a.street_number = random.randint(100, 9999)
            hotel_a.floor_number = random.randint(1, 20)
            hotel_a.building_age = random.randint(1, 50)
            hotel_a.renovation_year = random.randint(2000, 2024)
            hotel_a.hotel_chain_tier = random.choice(["Premium", "Standard", "Budget"])
            hotel_a.lobby_size_sqft = random.randint(500, 5000)
            hotel_a.employee_count = random.randint(10, 200)

            hotel_b.street_number = random.randint(100, 9999)
            hotel_b.floor_number = random.randint(1, 20)
            hotel_b.building_age = random.randint(1, 50)
            hotel_b.renovation_year = random.randint(2000, 2024)
            hotel_b.hotel_chain_tier = random.choice(["Premium", "Standard", "Budget"])
            hotel_b.lobby_size_sqft = random.randint(500, 5000)
            hotel_b.employee_count = random.randint(10, 200)

            tracking_a = {'correlation_mode': 'suppressed', 'utility': util_a,
                        'correlation_applied': False, 'correlation_direction': 'none',
                        'was_high_utility': util_a > 20,
                        'random_roll': None, 'threshold': self.spurious_correlation_strength,
                        'assignment_type': 'random'}
            tracking_b = {'correlation_mode': 'suppressed', 'utility': util_b,
                        'correlation_applied': False, 'correlation_direction': 'none',
                        'was_high_utility': util_b > 20,
                        'random_roll': None, 'threshold': self.spurious_correlation_strength,
                        'assignment_type': 'random'}
        else:
            # for normal and adversarial, use the existing method
            hotel_a, tracking_a = self.add_spurious_correlations(hotel_a, util_a, correlation_mode)
            hotel_b, tracking_b = self.add_spurious_correlations(hotel_b, util_b, correlation_mode)

        pair_tracking = {
            'hotel_a': tracking_a,
            'hotel_b': tracking_b,
            'both_correlated': tracking_a['correlation_applied'] and tracking_b['correlation_applied'],
            'neither_correlated': not tracking_a['correlation_applied'] and not tracking_b['correlation_applied'],
            'mixed_correlation': (tracking_a['correlation_applied'] != tracking_b['correlation_applied'])
        }

        return hotel_a, hotel_b, label, pair_tracking

    def modify_hotel_slightly(self, hotel: Hotel) -> Hotel:
        """Create a slightly modified version of a hotel"""
        new_hotel = copy.deepcopy(hotel)

        # randomly modify 1-2 features
        modifications = random.randint(1, 2)
        for _ in range(modifications):
            feature = random.choice(['price', 'distance', 'amenity', 'room_size'])
            if feature == 'price':
                new_hotel.price_per_night += random.randint(-20, 20)
                new_hotel.price_per_night = max(50, new_hotel.price_per_night)
            elif feature == 'distance':
                new_hotel.distance_to_convention_center += random.uniform(-0.5, 0.5)
                new_hotel.distance_to_convention_center = max(0.1, new_hotel.distance_to_convention_center)
            elif feature == 'amenity':
                new_hotel.has_pool = not new_hotel.has_pool
            elif feature == 'room_size':
                new_hotel.room_size_sqft += random.randint(-50, 50)
                new_hotel.room_size_sqft = max(150, new_hotel.room_size_sqft)

        return new_hotel

#@title 3. TIE CONSTRUCTION

# ============================================================
# function: apply spurious features at specific utility levels
# ============================================================
def apply_spurious_features_at_level(
    hotel: 'Hotel',
    utility_level: float,
    correlation_mode: str = "normal"
) -> 'Hotel':
    """
    Apply spurious features corresponding to a specific utility level.

    This is a helper function that manually sets spurious features
    without going through the probabilistic add_spurious_correlations method.

    Args:
        hotel: Hotel object to modify (will be modified in place)
        utility_level: Normalized utility level between 0.0 (worst) and 1.0 (best)
        correlation_mode: "normal" (high utility → good spurious) or
                         "adversarial" (high utility → bad spurious)

    Returns:
        Modified hotel object (same as input)
    """
    # For adversarial mode, invert the utility level
    if correlation_mode == "adversarial":
        utility_level = 1.0 - utility_level

    # Apply spurious features based on utility_level
    # These ranges match the original add_spurious_correlations logic

    # Street number: 100 to 9999
    street_range = 9999 - 100
    hotel.street_number = int(100 + utility_level * street_range)

    # Floor number: 1 to 20
    floor_range = 20 - 1
    hotel.floor_number = int(1 + utility_level * floor_range)

    # Building age: 1 to 50 (lower is better, so invert)
    age_range = 50 - 1
    hotel.building_age = int(50 - int(utility_level * age_range))

    # Renovation year: 2000 to 2024 (higher is better)
    hotel.renovation_year = int(2000 + int(utility_level * 24))

    # Chain tier: Budget < Standard < Premium
    if utility_level < 0.33:
        hotel.hotel_chain_tier = "Budget"
    elif utility_level < 0.67:
        hotel.hotel_chain_tier = "Standard"
    else:
        hotel.hotel_chain_tier = "Premium"

    # Lobby size: 500 to 5000 sqft
    lobby_range = 5000 - 500
    hotel.lobby_size_sqft = int(500 + int(utility_level * lobby_range))

    # Employee count: 10 to 200
    employee_range = 200 - 10
    hotel.employee_count = int(10 + int(utility_level * employee_range))

    return hotel


# ====================================================================
# function: decorrelate spurious feature in a given pair for tie cases
# ====================================================================
def decorrelate_spurious_features_in_pair(
    hotel_a: 'Hotel',
    hotel_b: 'Hotel',
    correlation_mode: str = "normal",
    strategy: str = "decorrelated_spurious",
    util_a: float = 0.0,  # <--- ADDED
    util_b: float = 0.0   # <--- ADDED
    ) -> Tuple['Hotel', 'Hotel', Dict]:
    """
    Decorrelate spurious features in a hotel pair (for tie cases).

    This function breaks the correlation between true utility and spurious features
    by assigning spurious features independently of the hotels' true utilities.

    Args:
        hotel_a: First hotel in pair
        hotel_b: Second hotel in pair
        correlation_mode: "normal" or "adversarial" (affects interpretation)
        strategy: How to assign spurious features:
            - "decorrelated_spurious": One hotel gets best spurious (1.0), one gets worst (0.0)
            - "random_uniform": Both get uniformly random spurious levels
            - "suppressed": Both get completely random spurious (no structure)

    Returns:
        Tuple of (modified hotel_a, modified hotel_b, tracking_dict)
    """
    tracking = {
        'strategy': strategy,
        'correlation_mode': correlation_mode
    }

    if strategy == "decorrelated_spurious":
        # Strategy 1: Maximize spurious margin while causal margin is small
        # One hotel gets the best spurious features, one gets the worst
        if random.random() < 0.5:
            # A gets good spurious, B gets bad
            apply_spurious_features_at_level(hotel_a, 1.0, correlation_mode)
            apply_spurious_features_at_level(hotel_b, 0.0, correlation_mode)
            tracking['spurious_assignment'] = 'A_max_B_min'
        else:
            # B gets good spurious, A gets bad
            apply_spurious_features_at_level(hotel_a, 0.0, correlation_mode)
            apply_spurious_features_at_level(hotel_b, 1.0, correlation_mode)
            tracking['spurious_assignment'] = 'A_min_B_max'

    elif strategy == "random_uniform":
        # Strategy 2: Both hotels get uniformly random spurious levels
        level_a = random.random()
        level_b = random.random()
        apply_spurious_features_at_level(hotel_a, level_a, correlation_mode)
        apply_spurious_features_at_level(hotel_b, level_b, correlation_mode)
        tracking['spurious_levels'] = {'A': level_a, 'B': level_b}

    elif strategy == "suppressed":
        # Strategy 3: Completely random spurious features (no correlation structure)
        # This matches the "suppressed" mode logic
        for hotel in [hotel_a, hotel_b]:
            hotel.street_number = random.randint(100, 9999)
            hotel.floor_number = random.randint(1, 20)
            hotel.building_age = random.randint(1, 50)
            hotel.renovation_year = random.randint(2000, 2024)
            hotel.hotel_chain_tier = random.choice(["Premium", "Standard", "Budget"])
            hotel.lobby_size_sqft = random.randint(500, 5000)
            hotel.employee_count = random.randint(10, 200)
        tracking['spurious_assignment'] = 'both_random'

    elif strategy == "standard_monotonic":
        # Strategy 4: FAILURE DEMONSTRATION
        # Apply spurious features strictly based on utility.
        # Since util_a approx util_b (Tie), features will be identical.
        # This results in Delta_phi_s = 0, so gradient = 0.
        # Another option is to inject noise, to avoid identical.

        # Normalize utility using the formula from HotelDatasetGenerator
        # (util + 10) / 50 clipped to [0, 1] [cite: 111]
        level_a = max(0, min(1, (util_a + 10) / 50.0))
        level_b = max(0, min(1, (util_b + 10) / 50.0))

        apply_spurious_features_at_level(hotel_a, level_a, correlation_mode)
        apply_spurious_features_at_level(hotel_b, level_b, correlation_mode)
        tracking['spurious_assignment'] = 'monotonic_identical'

    elif strategy == "standard_monotonic_noisy":
        # Monotonic assignment from utility, same as the original standard_monotonic case
        level_a = max(0.0, min(1.0, (util_a + 10.0) / 50.0))
        level_b = max(0.0, min(1.0, (util_b + 10.0) / 50.0))

        apply_spurious_features_at_level(hotel_a, level_a, correlation_mode)
        apply_spurious_features_at_level(hotel_b, level_b, correlation_mode)

        # Add small independent noise so spurious contrast is not exactly zero
        noise_scale = 0.05

        # Street number: valid range [100, 9999]
        hotel_a.street_number += int(random.gauss(0, noise_scale * (9999 - 100)))
        hotel_b.street_number += int(random.gauss(0, noise_scale * (9999 - 100)))
        hotel_a.street_number = max(100, min(9999, hotel_a.street_number))
        hotel_b.street_number = max(100, min(9999, hotel_b.street_number))

        # Floor number: valid range [1, 20]
        hotel_a.floor_number += int(random.gauss(0, noise_scale * (20 - 1)))
        hotel_b.floor_number += int(random.gauss(0, noise_scale * (20 - 1)))
        hotel_a.floor_number = max(1, min(20, hotel_a.floor_number))
        hotel_b.floor_number = max(1, min(20, hotel_b.floor_number))

        # Building age: valid range [1, 50]
        hotel_a.building_age += int(random.gauss(0, noise_scale * (50 - 1)))
        hotel_b.building_age += int(random.gauss(0, noise_scale * (50 - 1)))
        hotel_a.building_age = max(1, min(50, hotel_a.building_age))
        hotel_b.building_age = max(1, min(50, hotel_b.building_age))

        # Renovation year: valid range [2000, 2024]
        hotel_a.renovation_year += int(random.gauss(0, noise_scale * (2024 - 2000)))
        hotel_b.renovation_year += int(random.gauss(0, noise_scale * (2024 - 2000)))
        hotel_a.renovation_year = max(2000, min(2024, hotel_a.renovation_year))
        hotel_b.renovation_year = max(2000, min(2024, hotel_b.renovation_year))

        # Lobby size: valid range [500, 5000]
        hotel_a.lobby_size_sqft += int(random.gauss(0, noise_scale * (5000 - 500)))
        hotel_b.lobby_size_sqft += int(random.gauss(0, noise_scale * (5000 - 500)))
        hotel_a.lobby_size_sqft = max(500, min(5000, hotel_a.lobby_size_sqft))
        hotel_b.lobby_size_sqft = max(500, min(5000, hotel_b.lobby_size_sqft))

        # Employee count: valid range [10, 200]
        hotel_a.employee_count += int(random.gauss(0, noise_scale * (200 - 10)))
        hotel_b.employee_count += int(random.gauss(0, noise_scale * (200 - 10)))
        hotel_a.employee_count = max(10, min(200, hotel_a.employee_count))
        hotel_b.employee_count = max(10, min(200, hotel_b.employee_count))

        # Chain tier: keep unchanged to avoid awkward invalid categorical perturbations
        # If you want, you could randomize it occasionally, but leaving it fixed is cleaner.

        tracking["spurious_assignment"] = "standard_monotonic_noisy"
        tracking["noise_scale"] = noise_scale

    return hotel_a, hotel_b, tracking

#@title 4. DPO FORMATTING

# ============================
# function: Format DPO example (NEW VERSION)
# ============================
def format_dpo_example(
    context_prompt: str,
    option_a_text: str,
    option_b_text: str,
    util_a: float,
    util_b: float,
    label: PreferenceLabel,
    ) -> Optional[Dict]:
    """
    Formats the DPO example with the full context in the prompt
    and randomizes the order of A/B to ONE/TWO.
    """

    response_one = "Option ONE is the better choice."
    response_two = "Option TWO is the better choice."

    metadata = {}

    # Randomize position to avoid bias (A -> ONE or A -> TWO)
    if random.random() > 0.5:
        # Case 1: (A -> ONE), (B -> TWO)
        prompt_text_one = option_a_text
        prompt_text_two = option_b_text
        metadata['option_one_utility'] = util_a
        metadata['option_two_utility'] = util_b

        if label == PreferenceLabel.A_BETTER:
            chosen, rejected = response_one, response_two
            metadata['true_label'] = "A" # A = ONE
        else: # label == PreferenceLabel.B_BETTER
            chosen, rejected = response_two, response_one
            metadata['true_label'] = "B" # B = TWO
    else:
        # Case 2: (A -> TWO), (B -> ONE)
        prompt_text_one = option_b_text # Hotel B is in position ONE
        prompt_text_two = option_a_text # Hotel A is in position TWO
        metadata['option_one_utility'] = util_b
        metadata['option_two_utility'] = util_a

        if label == PreferenceLabel.A_BETTER: # Hotel A (now 'TWO') is better
            chosen, rejected = response_two, response_one
            metadata['true_label'] = "B" # B = TWO
        else: # label == PreferenceLabel.B_BETTER # Hotel B (now 'ONE') is better
            chosen, rejected = response_one, response_two
            metadata['true_label'] = "A" # A = ONE

    # --- New Prompt Format ---
    # Build the prompt *after* randomization
    full_prompt = (
        f"{context_prompt}\n\n"
        "Here are two options:\n\n"
        "--- Option ONE ---\n"
        f"{prompt_text_one}\n\n"
        "--- Option TWO ---\n"
        f"{prompt_text_two}\n\n"
        "--- Task ---\n"
        "Which of these two options is the better choice for the user?"
    )

    return {
        "prompt": full_prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": metadata # Contains true_label and correct utilities
    }
