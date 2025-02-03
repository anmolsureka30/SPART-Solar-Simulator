"""
Configuration file for Solar Powered Airship Flight Project
Contains all adjustable parameters for airship specifications, solar calculations,
flight parameters, and environmental factors.
"""

import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_create_directory(directory):
    """
    Safely create directories with robust error handling.
    
    Args:
        directory (Path or str): Directory path to create
    """
    try:
        # Convert to Path object if it's not already
        path = Path(directory)
        
        # Create directory with parents, exist_ok prevents errors if already exists
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {e}")

# =====================================
# Airship Physical Specifications
# =====================================
AIRSHIP_SPECS = {
    'length': 14.0,  # meters
    'ld_ratio': 3.05,  # length to diameter ratio
    'diameter': 4.6.0,  # meters
    'weight': 150.0,  # kg (empty weight)
    'max_payload': 10.0,  # kg
    'drag_coefficient': 0.03,  # dimensionless
    'max_altitude': 200.0,  # meters
    'min_altitude': 50.0,  # meters
    'surface_area': None,  # Will be calculated based on length and diameter
    'volume': None,  # Will be calculated based on length and diameter
}

# Calculate derived values
AIRSHIP_SPECS['surface_area'] = 3.14159 * AIRSHIP_SPECS['diameter'] * (
    AIRSHIP_SPECS['length'] - AIRSHIP_SPECS['diameter']) + \
    2 * (3.14159 * (AIRSHIP_SPECS['diameter']/2)**2)
AIRSHIP_SPECS['volume'] = (3.14159 * (AIRSHIP_SPECS['diameter']/2)**2 * 
                          AIRSHIP_SPECS['length'])

# =====================================
# Solar Panel Configuration
# =====================================
SOLAR_SPECS = {
    # Panel Layout
    'num_longitudinal_panels': 20,
    'num_circumferential_panels': 30,
    'panel_area': 50.0,  # Area of solar panels in square meters
    'panel_efficiency': 0.18,  # Base efficiency at standard conditions
    
    # Panel Performance
    'temp_coefficient': -0.0045,  # Efficiency change per °C (more accurate value)
    'reference_temperature': 25.0,  # °C (STC - Standard Test Conditions)
    'panel_thickness': 0.004,  # meters
    'panel_weight_per_m2': 2.5,  # kg/m²
    
    # Panel Orientation
    'tilt_angle_range': (-45, 45),  # degrees
    'minimum_sun_angle': 5.0,  # degrees (minimum angle for power generation)
    
    # Additional Parameters
    'soiling_factor': 0.97,  # Account for dust and dirt
    'wiring_efficiency': 0.98,  # Account for wiring losses
    'mismatch_factor': 0.98,  # Account for panel mismatch
    'spectral_factor': 0.97,  # Account for spectral variations
}

# =====================================
# Flight Parameters
# =====================================
FLIGHT_PARAMS = {
    'cruise_speed': 10.0,  # km/h
    'max_speed': 15.0,  # km/h
    'min_speed': 3.0,  # km/h
    'time_interval': 10,  # minutes (for calculations)
    'max_turning_radius': 500.0,  # meters
    'altitude_change_rate': 2.0,  # meters/second
    'safety_margin': 1.2,  # Safety factor for flight calculations
}

# =====================================
# Environmental Factors
# =====================================
ENVIRONMENT = {
    # Cloud Effects on Solar Radiation
    'clear_sky_factor': 1.0,
    'light_clouds_factor': 0.8,  # Adjusted for more realistic attenuation
    'medium_clouds_factor': 0.5,  # Adjusted for more realistic attenuation
    'heavy_clouds_factor': 0.2,  # Adjusted for more realistic attenuation
    
    # Wind Limits
    'max_headwind': 40.0,  # km/h
    'max_crosswind': 25.0,  # km/h
    'wind_safety_margin': 1.2,
    
    # Atmospheric Properties
    'air_density_sea_level': 1.225,  # kg/m³
    'temperature_lapse_rate': -0.0065,  # °C/m
    'pressure_sea_level': 1013.25,  # hPa
    'gravity': 9.81,  # m/s²
    
    # Weather Conditions
    'max_temperature': 45.0,  # °C
    'min_temperature': -10.0,  # °C
    'max_humidity': 100.0,  # %
    'min_humidity': 0.0,  # %
}

# =====================================
# Power System
# =====================================
POWER_SYSTEM = {
    'battery_capacity': 500.0,  # kWh
    'battery_max_discharge': 0.8,  # maximum depth of discharge
    'battery_charge_efficiency': 0.95,
    'inverter_efficiency': 0.97,
    'mppt_efficiency': 0.98,  # Maximum Power Point Tracking efficiency
    'power_margin': 1.2,  # safety factor for power calculations
    'auxiliary_power_draw': 2.0,  # kW (for systems excluding propulsion)
    'battery_weight_per_kwh': 7.0,  # kg/kWh (modern Li-ion batteries)
    'min_state_of_charge': 0.2,  # Minimum allowable battery state of charge
}

# =====================================
# API Configuration
# =====================================
API_CONFIG = {
    'weather_api_key': "7f59ae0ebf513f9439648ca158e22e31",
    'openweather_base_url': "https://api.openweathermap.org/data/2.5/weather",
    'max_api_calls_per_minute': 60,
    'retry_attempts': 5,  # Increased for better reliability
    'timeout': 30,  # seconds
    'ssl_verify': True,  # Enable SSL verification
    'cache_duration': 3600,  # seconds (1 hour) to cache weather data
}

# =====================================
# File Paths and Directories
# =====================================
BASE_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

PATHS = {
    'base_dir': BASE_DIR,
    'weather_data': BASE_DIR / "weather_data",
    'irradiance_data': BASE_DIR / "irradiance_data",
    'logs': BASE_DIR / "logs",
    'waypoints': BASE_DIR / "waypoints.csv",
    'airship_geometry': BASE_DIR / "airship_geometry.json",
    'visualizations': OUTPUT_DIR / "visualizations",
    'power_output': OUTPUT_DIR / "power_output.csv",
    'simulation_results': OUTPUT_DIR / "simulation_results",
    'cache': BASE_DIR / "cache"
}

# =====================================
# Visualization Settings
# =====================================
VISUALIZATION = {
    'map_style': 'satellite',
    'route_color': '#FF4444',
    'power_graph_color': '#2196F3',
    'altitude_graph_color': '#4CAF50',
    'chart_dpi': 300,
    'plot_size': (12, 8),  # inches
    'font_family': 'Arial',
    'title_size': 14,
    'label_size': 12,
    'tick_size': 10,
    'grid_alpha': 0.3,
}

# Updated directory creation section
DIRECTORIES_TO_CREATE = [
    BASE_DIR,
    OUTPUT_DIR,
    PATHS['weather_data'],
    PATHS['irradiance_data'],
    PATHS['logs'],
    PATHS['visualizations'],
    PATHS['simulation_results'],
    PATHS['cache']
]

# Use safe directory creation
for directory in DIRECTORIES_TO_CREATE:
    safe_create_directory(directory)
