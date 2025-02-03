import math
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class NavigationParameters:
    """Configuration parameters for flight path planning."""
    cruise_speed_kmh: float  # Cruise speed in km/h
    max_speed_kmh: float     # Maximum speed in km/h
    min_speed_kmh: float     # Minimum safe speed in km/h
    interval_minutes: float  # Time interval between waypoints
    altitude_m: float        # Cruise altitude in meters
    max_wind_speed_kmh: float  # Maximum safe wind speed
    reserve_factor: float      # Safety reserve factor (typically 1.2-1.5)


class AirshipNavigator:
    EARTH_RADIUS_KM = 6371
    MIN_ROUTE_SEGMENTS = 50  # Minimum segments for long routes

    @staticmethod
    def calculate_great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance with improved precision."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return AirshipNavigator.EARTH_RADIUS_KM * c

    @staticmethod
    def calculate_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        initial_bearing = math.atan2(y, x)
        return (math.degrees(initial_bearing) + 360) % 360

    @staticmethod
    def destination_point(lat: float, lon: float, distance: float, bearing: float, altitude_m: float) -> Tuple[float, float]:
        """Calculate destination point considering altitude."""
        R = AirshipNavigator.EARTH_RADIUS_KM + (altitude_m / 1000)
        lat, lon = map(math.radians, [lat, lon])
        bearing = math.radians(bearing)
        angular_distance = distance / R

        lat2 = math.asin(
            math.sin(lat) * math.cos(angular_distance) +
            math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
        )

        lon2 = lon + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
            math.cos(angular_distance) - math.sin(lat) * math.sin(lat2)
        )

        return math.degrees(lat2), math.degrees(lon2)


def generate_waypoints(params: NavigationParameters, start_coords: Tuple[float, float],
                       end_coords: Tuple[float, float], max_altitude: float = None,
                       start_time: datetime = None) -> List[Dict]:
    """
    Generate waypoints for a flight path based on navigation parameters.

    Args:
        params: NavigationParameters instance
        start_coords: Tuple with starting latitude and longitude
        end_coords: Tuple with ending latitude and longitude
        max_altitude: Maximum allowable altitude for the flight path (optional)
        start_time: Start time of the flight (optional, defaults to current UTC time)

    Returns:
        List of dictionaries containing waypoint information
    """
    navigator = AirshipNavigator()

    # Calculate total distance and initial bearing
    distance_km = navigator.calculate_great_circle_distance(*start_coords, *end_coords)
    initial_bearing = navigator.calculate_initial_bearing(*start_coords, *end_coords)

    # Calculate number of waypoints based on flight duration and interval
    flight_hours = distance_km / params.cruise_speed_kmh
    num_segments = max(
        navigator.MIN_ROUTE_SEGMENTS,
        int((flight_hours * 60) / params.interval_minutes)
    )

    step_distance = distance_km / num_segments
    waypoints = []
    current_point = start_coords
    elapsed_distance = 0
    current_time = start_time or datetime.utcnow()

    for i in range(num_segments + 1):
        current_bearing = navigator.calculate_initial_bearing(*current_point, *end_coords)
        waypoint = {
            "index": i,
            "latitude": current_point[0],
            "longitude": current_point[1],
            "distance_km": round(elapsed_distance, 2),
            "bearing": round(current_bearing, 2),
            "altitude_m": min(params.altitude_m, max_altitude) if max_altitude else params.altitude_m,
            "planned_speed_kmh": params.cruise_speed_kmh,
            "elapsed_hours": round(elapsed_distance / params.cruise_speed_kmh, 2),
            "remaining_km": round(distance_km - elapsed_distance, 2),
            "completion_percentage": round((elapsed_distance / distance_km) * 100, 2),
            "timestamp": current_time  # Keep timestamp as datetime object
        }
        waypoints.append(waypoint)

        if i < num_segments:
            current_point = navigator.destination_point(
                *current_point, step_distance, current_bearing, waypoint["altitude_m"]
            )
            elapsed_distance += step_distance
            current_time += timedelta(minutes=params.interval_minutes)  # Increment timestamp

    return waypoints



def save_waypoints_to_csv(waypoints: List[Dict], output_file: str) -> None:
    """
    Save waypoints to a CSV file.

    Args:
        waypoints: List of waypoint dictionaries
        output_file: Path to output CSV file
    """
    df = pd.DataFrame(waypoints)
    df.to_csv(output_file, index=False)


# Example usage for Springbok to Natal route
if __name__ == "__main__":
    SPRINGBOK = (-29.6643, 17.8778)
    NATAL = (-5.7793, -35.2009)

    params = NavigationParameters(
        cruise_speed_kmh=80.0,
        max_speed_kmh=100.0,
        min_speed_kmh=40.0,
        interval_minutes=30.0,
        altitude_m=1000.0,
        max_wind_speed_kmh=40.0,
        reserve_factor=1.3
    )

    waypoints = generate_waypoints(params, SPRINGBOK, NATAL, start_time=datetime.utcnow())
    save_waypoints_to_csv(waypoints, "springbok_natal_route.csv")

    print(f"Generated {len(waypoints)} waypoints. Waypoints saved to springbok_natal_route.csv.")
