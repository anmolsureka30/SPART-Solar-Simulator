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
import solar_calculation  # Updated import

class SolarAirshipSimulation:
    """Manages the complete simulation pipeline for the solar-powered airship."""

    def __init__(self):
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
        # Initialize weather fetcher with corrected configuration
        weather_config = WeatherAPIConfig(
            openweather_api_key=API_CONFIG['weather_api_key']
        )
        self.weather_fetcher = WeatherDataFetcher(weather_config)
        self.power_calculator = PowerCalculator()
        self.results: Dict[str, Any] = {}
        self.batch_size = 20

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

    def create_output_directories(self):
        """Create necessary output directories."""
        for path_key, path in PATHS.items():
            try:
                if isinstance(path, (str, Path)):
                    dir_path = Path(path)
                    if dir_path.suffix == '':  # Only create directories, not files
                        dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not create directory for {path_key}: {e}")

    def validate_inputs(self, start_point: Tuple[float, float], end_point: Tuple[float, float]):
        """Validate input parameters."""
        for point, point_name in [(start_point, "start"), (end_point, "end")]:
            lat, lon = point
            if not (-90 <= lat <= 90):
                raise ValueError(f"Invalid latitude in {point_name} point: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Invalid longitude in {point_name} point: {lon}")

        if not API_CONFIG.get('weather_api_key'):
            raise ValueError("Weather API key not configured")
            
        if FLIGHT_PARAMS['cruise_speed'] > FLIGHT_PARAMS['max_speed']:
            raise ValueError("Cruise speed cannot exceed maximum speed")
        if FLIGHT_PARAMS['min_speed'] > FLIGHT_PARAMS['cruise_speed']:
            raise ValueError("Minimum speed cannot exceed cruise speed")

    def _process_weather_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of waypoints for weather data."""
        weather_data = []
        
        for _, row in batch_df.iterrows():
            try:
                data = self.weather_fetcher.fetch_weather_data(
                    lat=row['latitude'],
                    lon=row['longitude'],
                    timestamp=row['timestamp'].to_pydatetime()
                )
                data.update({
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'timestamp': row['timestamp']
                })
                weather_data.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to fetch weather data: {e}")
                fallback = self.weather_fetcher._get_fallback_data(
                    row['latitude'],
                    row['longitude'],
                    row['timestamp'].to_pydatetime()
                )
                fallback.update({
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'timestamp': row['timestamp']
                })
                weather_data.append(fallback)
                
        return weather_data

    def fetch_weather_data(self, waypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch and integrate weather data into waypoints."""
        self.logger.info("Fetching weather data...")

        df = waypoints_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        all_weather_data = []
        total_batches = len(df) // self.batch_size + (1 if len(df) % self.batch_size else 0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._process_weather_batch, df.iloc[i:i + self.batch_size])
                for i in range(0, len(df), self.batch_size)
            ]
                
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    batch_results = future.result()
                    all_weather_data.extend(batch_results)
                    self.logger.info(f"Completed batch {idx + 1}/{total_batches}")
                except Exception as e:
                    self.logger.error(f"Error processing batch {idx + 1}: {e}")

        weather_df = pd.DataFrame(all_weather_data)
        result_df = pd.merge(
            df,
            weather_df,
            on=['latitude', 'longitude', 'timestamp'],
            how='left'
        )
        
        return result_df

    def generate_flight_path(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> pd.DataFrame:
        """Generate and save flight path waypoints."""
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

        waypoints = generate_waypoints(
            params=params,
            start_coords=start_point,
            end_coords=end_point,
        )

        waypoints_df = pd.DataFrame(waypoints)
        initial_time = datetime.now().replace(second=0, microsecond=0)
        
        waypoints_df['timestamp'] = [
            initial_time + timedelta(minutes=i * FLIGHT_PARAMS['time_interval'])
            for i in range(len(waypoints_df))
        ]

        try:
            Path(PATHS['waypoints']).parent.mkdir(parents=True, exist_ok=True)
            waypoints_df.to_csv(PATHS['waypoints'], index=False)
            self.results['waypoints'] = len(waypoints_df)
        except Exception as e:
            self.logger.error(f"Error saving waypoints: {e}")
        
        self.logger.info(f"Generated {len(waypoints_df)} waypoints")
        return waypoints_df

    def calculate_power_generation(self, waypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate power generation with improved error handling."""
        self.logger.info("Calculating power generation...")
        
        try:
            waypoints_df['timestamp'] = pd.to_datetime(waypoints_df['timestamp'])
            waypoints_with_weather = self.fetch_weather_data(waypoints_df)
            
            power_df = self.power_calculator.calculate_power(waypoints_with_weather)
            
            try:
                Path(PATHS['power_output']).parent.mkdir(parents=True, exist_ok=True)
                power_df.to_csv(PATHS['power_output'], index=False)
            except Exception as e:
                self.logger.error(f"Error saving power calculations: {e}")
            
            return power_df
            
        except Exception as e:
            self.logger.error(f"Error in power generation calculations: {e}")
            raise

    def run_simulation(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """Run the complete simulation with comprehensive error handling."""
        self.logger.info(f"Starting solar airship simulation {self.simulation_id}...")
        
        try:
            self.validate_inputs(start_point, end_point)
            self.create_output_directories()

            waypoints_df = self.generate_flight_path(start_point, end_point)
            power_df = self.calculate_power_generation(waypoints_df)
            
            self.logger.info("Simulation completed successfully")
            return power_df
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise

def main():
    """Main entry point for the simulation."""
    start_point = (-29.6667, 17.8833)  # Starting point coordinates
    end_point = (-4.3155, -38.8223)    # Ending point coordinates
    
    try:
        simulation = SolarAirshipSimulation()
        result_df = simulation.run_simulation(start_point, end_point)
        
        if result_df is not None and not result_df.empty:
            print("\nSimulation Results Summary:")
            print(f"Total waypoints: {len(result_df)}")
            print("\nPower Generation Summary:")
            print(result_df['total_power'].describe())
            
    except Exception as e:
        logging.error(f"Fatal error in simulation: {e}")
        raise

if __name__ == "__main__":
    main()