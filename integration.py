import pandas as pd
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import concurrent.futures
import numpy as np

from config import (
    API_CONFIG, PATHS, AIRSHIP_SPECS,
    FLIGHT_PARAMS, SOLAR_SPECS, VISUALIZATION, ENVIRONMENT
)
from flight_path import NavigationParameters, generate_waypoints
from weather_data_fetcher import WeatherAPIConfig, WeatherDataFetcher
from airship_geometry import AirshipParameters, AirshipGeometry
from power_calculation import PowerCalculator

class SolarAirshipSimulation:
    """Manages the complete simulation pipeline for the solar-powered airship."""

    def __init__(self):
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
        weather_config = WeatherAPIConfig(
            openweather_api_key=API_CONFIG['weather_api_key']
        )
        self.weather_fetcher = WeatherDataFetcher(weather_config)
        self.power_calculator = PowerCalculator()
        self.results: Dict[str, Any] = {}
        self.batch_size = 20
        self.weather_data_file = Path(PATHS['weather_data']) / "weather_data.csv"

    def setup_logging(self):
        """Configure logging for the simulation."""
        log_file = Path(PATHS['logs']) / f"simulation_{self.simulation_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _process_weather_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Fetch weather data for a batch of waypoints."""
        weather_data = []
        
        for _, row in batch_df.iterrows():
            try:
                data = self.weather_fetcher.fetch_weather_data(
                    lat=row['latitude'],
                    lon=row['longitude'],
                    timestamp=row['timestamp'].to_pydatetime()
                )
            except Exception as e:
                self.logger.warning(f"Failed to fetch weather data at ({row['latitude']}, {row['longitude']}): {e}")
                data = self.weather_fetcher._get_fallback_data(
                    row['latitude'],
                    row['longitude'],
                    row['timestamp'].to_pydatetime()
                )
            data.update({
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'timestamp': row['timestamp']
            })
            weather_data.append(data)
        
        return weather_data

    def fetch_weather_data(self, waypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data and save to CSV."""
        self.logger.info("Fetching weather data...")
        df = waypoints_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        all_weather_data = []
        total_batches = len(df) // self.batch_size + (1 if len(df) % self.batch_size else 0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._process_weather_batch, df.iloc[i:i + self.batch_size]): i // self.batch_size + 1
                for i in range(0, len(df), self.batch_size)
            }
            
            for future in concurrent.futures.as_completed(futures):
                batch_num = futures[future]
                try:
                    batch_results = future.result()
                    all_weather_data.extend(batch_results)
                    self.logger.info(f"Completed batch {batch_num}/{total_batches}")
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_num}: {e}")
        
        weather_df = pd.DataFrame(all_weather_data)
        result_df = pd.merge_asof(df.sort_values('timestamp'), weather_df.sort_values('timestamp'),
                                  on='timestamp', by=['latitude', 'longitude'], direction='nearest')
        
        try:
            self.weather_data_file.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(self.weather_data_file, index=False)
            self.logger.info(f"Weather data saved to {self.weather_data_file}")
        except Exception as e:
            self.logger.error(f"Error saving weather data: {e}")
        
        return result_df

    def generate_flight_path(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> pd.DataFrame:
        """Generate flight path waypoints."""
        self.logger.info("Generating flight path waypoints...")
        params = NavigationParameters(
            cruise_speed_kmh=FLIGHT_PARAMS['cruise_speed'],
            max_speed_kmh=FLIGHT_PARAMS['max_speed'],
            min_speed_kmh=FLIGHT_PARAMS['min_speed'],
            interval_minutes=FLIGHT_PARAMS['time_interval'],
            altitude_m=AIRSHIP_SPECS['max_altitude'],
            max_wind_speed_kmh=ENVIRONMENT['max_headwind'],
            reserve_factor=1.3
        )
        waypoints = generate_waypoints(params, start_coords=start_point, end_coords=end_point)
        waypoints_df = pd.DataFrame(waypoints)
        waypoints_df['timestamp'] = [
            datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=i * FLIGHT_PARAMS['time_interval'])
            for i in range(len(waypoints_df))
        ]
        return waypoints_df

    def run_simulation(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """Run the complete simulation."""
        self.logger.info(f"Starting simulation {self.simulation_id}...")
        try:
            waypoints_df = self.generate_flight_path(start_point, end_point)
            weather_data_df = self.fetch_weather_data(waypoints_df)
            self.logger.info("Simulation completed successfully")
            return weather_data_df
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise

if __name__ == "__main__":
    simulation = SolarAirshipSimulation()
    start_point = (-29.6667, 17.8833)
    end_point = (-4.3155, -38.8223)
    simulation.run_simulation(start_point, end_point)
