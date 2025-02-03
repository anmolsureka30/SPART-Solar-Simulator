import pvlib
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict
from dataclasses import dataclass
import math
import os
import logging
from pathlib import Path


@dataclass
class AtmosphericParameters:
    """Atmospheric condition parameters for irradiance and air mass calculations."""
    pressure: float = 101325.0  # Standard pressure at sea level (Pa)
    altitude: float = 0.0       # Altitude in meters
    temperature: float = 25.0   # Ambient temperature (°C)
    relative_humidity: float = 50.0  # Relative humidity (%)
    albedo: float = 0.2         # Ground reflectance (fraction)


@dataclass
class SolarPanelSpecs:
    """Specifications of solar panels for power output calculations."""
    efficiency: float  # Panel efficiency (0-1)
    temp_coefficient: float  # Temperature coefficient (%/°C)
    reference_temperature: float = 25.0  # Reference temperature (°C)


def safe_create_directory(directory_path):
    """Safely create directories, handling different path types."""
    try:
        # Convert to string if it's a Path object
        if hasattr(directory_path, 'as_posix'):
            directory_path = directory_path.as_posix()
        
        # Ensure it's a string path
        directory_path = str(directory_path)
        
        # Create directory
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        logging.warning(f"Could not create directory {directory_path}: {e}")


def calculate_solar_position(latitude: float, longitude: float, timestamp: datetime, altitude: float = 0.0) -> Dict[str, float]:
    """Calculate solar position parameters with robust error handling."""
    try:
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        pressure = 101325 * np.exp(-altitude / 8500)  # Adjust pressure for altitude
        solar_position = pvlib.solarposition.get_solarposition(
            time=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            pressure=pressure,
            method='nrel_numpy'
        )
        return {
            "zenith": float(solar_position["zenith"].iloc[0]),
            "azimuth": float(solar_position["azimuth"].iloc[0]),
            "elevation": float(solar_position["elevation"].iloc[0]),
            "apparent_zenith": float(solar_position["apparent_zenith"].iloc[0])
        }
    except Exception as e:
        logging.error(f"Error in calculate_solar_position: {e}")
        raise RuntimeError(f"Solar position calculation failed: {e}")


def calculate_clearsky_irradiance(latitude: float, longitude: float, timestamp: datetime, atmospheric_params: AtmosphericParameters) -> Dict[str, float]:
    """Calculate clear-sky irradiance with robust error handling."""
    try:
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        location = pvlib.location.Location(
            latitude, 
            longitude, 
            altitude=atmospheric_params.altitude
        )
        clearsky = location.get_clearsky(timestamp)
        return {
            "ghi": float(clearsky["ghi"].iloc[0]),
            "dni": float(clearsky["dni"].iloc[0]),
            "dhi": float(clearsky["dhi"].iloc[0])
        }
    except Exception as e:
        logging.error(f"Error in calculate_clearsky_irradiance: {e}")
        raise RuntimeError(f"Clearsky irradiance calculation failed: {e}")


def calculate_incident_irradiance(
    normal_vector: List[float], 
    solar_position: Dict[str, float], 
    irradiance: Dict[str, float], 
    atmospheric_params: AtmosphericParameters
) -> float:
    """Calculate incident irradiance on a tilted surface with enhanced error handling."""
    try:
        normal = np.array(normal_vector) / np.linalg.norm(normal_vector)
        solar_vector = np.array([
            np.sin(np.radians(solar_position["zenith"])) * np.cos(np.radians(solar_position["azimuth"])),
            np.sin(np.radians(solar_position["zenith"])) * np.sin(np.radians(solar_position["azimuth"])),
            np.cos(np.radians(solar_position["zenith"]))
        ])
        
        incident_angle = np.arccos(np.clip(np.dot(normal, solar_vector), -1.0, 1.0))
        
        direct = irradiance["dni"] * max(np.cos(incident_angle), 0)
        sky_diffuse = pvlib.irradiance.isotropic(
            surface_tilt=np.degrees(np.arccos(normal[2])),
            dhi=irradiance["dhi"]
        )
        ground_reflected = irradiance["ghi"] * atmospheric_params.albedo * (1 - normal[2]) / 2
        
        return direct + sky_diffuse + ground_reflected
    except Exception as e:
        logging.error(f"Error in calculate_incident_irradiance: {e}")
        raise RuntimeError(f"Incident irradiance calculation failed: {e}")


def calculate_panel_output(
    incident_irradiance: float, 
    temperature: float, 
    panel_specs: SolarPanelSpecs
) -> float:
    """Calculate power output of the solar panel considering temperature effects."""
    try:
        temp_effect = 1 + panel_specs.temp_coefficient * (temperature - panel_specs.reference_temperature) / 100
        return max(0.0, incident_irradiance * panel_specs.efficiency * temp_effect)
    except Exception as e:
        logging.error(f"Error in calculate_panel_output: {e}")
        raise RuntimeError(f"Panel output calculation failed: {e}")


class SolarCalculator:
    """A robust calculator class for solar energy computations."""
    def __init__(self, panel_specs: SolarPanelSpecs):
        self.panel_specs = panel_specs

    def calculate_solar_position(self, latitude, longitude, timestamp, altitude=0.0):
        return calculate_solar_position(latitude, longitude, timestamp, altitude)

    def calculate_clearsky_irradiance(self, latitude, longitude, timestamp, atmospheric_params):
        return calculate_clearsky_irradiance(latitude, longitude, timestamp, atmospheric_params)

    def calculate_incident_irradiance(self, normal_vector, solar_position, irradiance, atmospheric_params):
        return calculate_incident_irradiance(normal_vector, solar_position, irradiance, atmospheric_params)

    def calculate_panel_output(self, incident_irradiance, temperature):
        return calculate_panel_output(incident_irradiance, temperature, self.panel_specs)


def calculate_total_power(
    solar_calculator: SolarCalculator,
    panels: List[Dict],
    latitude: float,
    longitude: float,
    timestamp: datetime,
    atmospheric_params: AtmosphericParameters
) -> Dict[str, float]:
    """Calculate total power output for multiple panels with comprehensive error handling."""
    try:
        solar_position = solar_calculator.calculate_solar_position(
            latitude, longitude, timestamp, atmospheric_params.altitude
        )
        
        irradiance = solar_calculator.calculate_clearsky_irradiance(
            latitude, longitude, timestamp, atmospheric_params
        )

        total_power = 0.0
        panel_outputs = []

        for panel in panels:
            try:
                incident_irradiance = solar_calculator.calculate_incident_irradiance(
                    normal_vector=panel.get("normal_vector", [0, 0, 1]),
                    solar_position=solar_position,
                    irradiance=irradiance,
                    atmospheric_params=atmospheric_params
                )
                power_output = solar_calculator.calculate_panel_output(
                    incident_irradiance,
                    atmospheric_params.temperature
                )
                total_power += power_output * panel.get("area", 1.0)
                panel_outputs.append(power_output)
            except Exception as panel_error:
                logging.error(f"Error processing panel: {panel_error}")

        return {
            "total_power": total_power,
            "average_panel_output": np.mean(panel_outputs) if panel_outputs else 0.0,
            "max_panel_output": max(panel_outputs) if panel_outputs else 0.0,
            "min_panel_output": min(panel_outputs) if panel_outputs else 0.0,
            "solar_position": solar_position,
            "irradiance": irradiance
        }
    except Exception as e:
        logging.error(f"Total power calculation failed: {e}")
        raise RuntimeError(f"Comprehensive power calculation error: {e}")


# Example usage and demonstration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    panel_specs = SolarPanelSpecs(
        efficiency=0.20,
        temp_coefficient=-0.4
    )
    calculator = SolarCalculator(panel_specs)
    
    atmospheric_params = AtmosphericParameters(
        altitude=1000.0,
        temperature=20.0,
        relative_humidity=60.0,
        albedo=0.2
    )
    
    panels = [
        {"normal_vector": [0, 0, 1], "area": 2.0},
        {"normal_vector": [0.5, 0.5, 0.707], "area": 1.5},
    ]
    
    power_summary = calculate_total_power(
        calculator,
        panels,
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=datetime.now(timezone.utc),
        atmospheric_params=atmospheric_params
    )
    
    print("Power Output Summary:")
    for key, value in power_summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue:.2f}")
        else:
            print(f"{key}: {value:.2f}")