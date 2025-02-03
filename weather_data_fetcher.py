import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import ssl
import math

@dataclass
class WeatherAPIConfig:
    """Configuration for weather API services."""
    openweather_api_key: str
    OPENWEATHER_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    max_retries: int = 3
    timeout: int = 30
    ssl_verify: bool = True
    solar_api_key: Optional[str] = None
    backup_api_key: Optional[str] = None

@dataclass
class SolarPosition:
    azimuth: float
    elevation: float
    zenith: float

class CustomHTTPAdapter(HTTPAdapter):
    """Custom HTTP Adapter with SSL configuration."""
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.ssl_context or ssl.create_default_context()
        return super().init_poolmanager(*args, **kwargs)

class SolarIrradianceCalculator:
    def __init__(self):
        self.solar_constant = 1361.0  # Solar constant in W/m²

    def calculate_solar_position(self, latitude: float, longitude: float, timestamp: datetime = None) -> Tuple[float, float]:
        """Calculate solar position (elevation and azimuth) for given coordinates and time."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        day_of_year = timestamp.timetuple().tm_yday

        declination = math.radians(23.45 * math.sin(math.radians((360/365) * (day_of_year - 81))))
        hour = timestamp.hour + timestamp.minute/60 + timestamp.second/3600
        hour_angle = math.radians(15 * (hour - 12))

        elevation_rad = math.asin(
            math.sin(lat_rad) * math.sin(declination) +
            math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        
        azimuth_rad = math.atan2(
            -math.cos(declination) * math.sin(hour_angle),
            math.cos(lat_rad) * math.sin(declination) -
            math.sin(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )

        elevation_deg = math.degrees(elevation_rad)
        azimuth_deg = (math.degrees(azimuth_rad) + 180) % 360

        return elevation_deg, azimuth_deg

    def calculate_irradiance(self, elevation: float, cloud_cover: float) -> Dict[str, float]:
        """Calculate solar irradiance components based on elevation angle and cloud cover."""
        if elevation <= 0:
            return {"DNI": 0.0, "DHI": 0.0, "GHI": 0.0}

        # Calculate air mass
        zenith = 90 - elevation
        air_mass = 1 / math.cos(math.radians(zenith))
        
        # Calculate clear sky DNI
        dni_clear = self.solar_constant * (0.7 ** air_mass)
        
        # Apply cloud cover effect (0-100%)
        cloud_factor = 1 - (cloud_cover / 100) * 0.75
        dni = dni_clear * cloud_factor

        # Calculate DHI (diffuse horizontal irradiance)
        dhi = dni_clear * (1 - cloud_factor) * 0.5

        # Calculate GHI (global horizontal irradiance)
        ghi = dni * math.sin(math.radians(elevation)) + dhi

        return {
            "DNI": dni,
            "DHI": dhi,
            "GHI": ghi
        }


class WeatherDataFetcher:
    """Fetch and process weather data from APIs with improved error handling."""

    def __init__(self, config: WeatherAPIConfig):
        self.config = config
        self.solar_calculator = SolarIrradianceCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Configure session with retry strategy
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and SSL configuration."""
        session = requests.Session()
        
        ssl_context = ssl.create_default_context()
        ssl_context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = CustomHTTPAdapter(
            ssl_context=ssl_context,
            max_retries=retry_strategy
        )
        
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session

    def fetch_weather_data(self, lat: float, lon: float, timestamp: datetime) -> Dict[str, float]:
        """Fetch weather and solar data for a single point."""
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            elif not isinstance(timestamp, datetime):
                raise ValueError("Timestamp must be a datetime object.")

            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.config.openweather_api_key,
                "units": "metric"
            }
            
            response = self.session.get(
                self.config.OPENWEATHER_URL,
                params=params,
                timeout=self.config.timeout,
                verify=self.config.ssl_verify
            )
            response.raise_for_status()
            weather_data = response.json()
            
            elevation, azimuth = self.solar_calculator.calculate_solar_position(lat, lon, timestamp)
            cloud_cover = weather_data.get('clouds', {}).get('all', 50)
            irradiance = self.solar_calculator.calculate_irradiance(elevation, cloud_cover)

            return {
                **irradiance,
                "temperature": weather_data.get('main', {}).get('temp', 25),
                "pressure": weather_data.get('main', {}).get('pressure', 1013),
                "humidity": weather_data.get('main', {}).get('humidity', 50),
                "cloud_cover": cloud_cover,
                "wind_speed": weather_data.get('wind', {}).get('speed', 0),
                "wind_direction": weather_data.get('wind', {}).get('deg', 0),
                "solar_elevation": elevation,
                "solar_azimuth": azimuth
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {e}")
            return self._get_fallback_data(lat, lon, timestamp)

    def _get_fallback_data(self, lat: float, lon: float, timestamp: datetime) -> Dict[str, float]:
        """Generate fallback weather data when API requests fail."""
        elevation, azimuth = self.solar_calculator.calculate_solar_position(lat, lon, timestamp)
        irradiance = self.solar_calculator.calculate_irradiance(elevation, 50)
        self.logger.warning(f"Using fallback weather data for lat: {lat}, lon: {lon}")
        
        return {
            **irradiance,
            "temperature": 25,
            "pressure": 1013,
            "humidity": 50,
            "cloud_cover": 50,
            "wind_speed": 0,
            "wind_direction": 0,
            "solar_elevation": elevation,
            "solar_azimuth": azimuth
        }

    def fetch_weather_for_dataframe(self, waypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data for all waypoints in a DataFrame."""
        self.logger.info("Fetching weather data for waypoints...")

        waypoints_df['timestamp'] = pd.to_datetime(waypoints_df['timestamp'])
        max_workers = min(10, len(waypoints_df))
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in waypoints_df.iterrows():
                futures.append(
                    executor.submit(
                        self.fetch_weather_data,
                        lat=row['latitude'],
                        lon=row['longitude'],
                        timestamp=row['timestamp'].to_pydatetime()
                    )
                )
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to fetch weather data: {e}")
                    results.append(self._get_fallback_data(
                        row['latitude'],
                        row['longitude'],
                        row['timestamp'].to_pydatetime()
                    ))

        if not results:
            self.logger.error("No valid weather data fetched. Returning an empty DataFrame.")
            return pd.DataFrame()

        weather_df = pd.DataFrame(results)
        self.logger.info(f"Successfully fetched weather data for {len(results)} waypoints.")
        return weather_df