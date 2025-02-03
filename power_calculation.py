import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from airship_geometry import AirshipGeometry, AirshipParameters
from solar_calculation import calculate_solar_position, calculate_incident_irradiance
from config import SOLAR_SPECS, ENVIRONMENT, POWER_SYSTEM, AIRSHIP_SPECS

class PowerCalculator:
    """
    Handles all power-related calculations for the solar airship.
    """
    
    def __init__(self):
        self.panel_area = self._calculate_panel_area()
        self.total_system_losses = self._calculate_system_losses()
        
    def _calculate_panel_area(self) -> float:
        """Calculate the area of each solar panel based on airship geometry."""
        longitudinal_spacing = AIRSHIP_SPECS['length'] / SOLAR_SPECS.get('num_longitudinal_panels', 20)
        circumferential_spacing = np.pi * AIRSHIP_SPECS['diameter'] / SOLAR_SPECS.get('num_circumferential_panels', 30)
        return longitudinal_spacing * circumferential_spacing

    def _calculate_system_losses(self) -> float:
        """Calculate total system losses from various efficiency factors."""
        return (1 - POWER_SYSTEM.get('inverter_efficiency', 0.97)) + \
               (1 - POWER_SYSTEM.get('battery_charge_efficiency', 0.95))

    def _adjust_efficiency_for_temperature(self, temperature: float) -> float:
        """Adjust solar panel efficiency based on temperature."""
        temp_difference = temperature - SOLAR_SPECS.get('reference_temperature', 25)
        efficiency_adjustment = SOLAR_SPECS.get('temp_coefficient', -0.0045) * temp_difference
        return SOLAR_SPECS.get('panel_efficiency', 0.18) + efficiency_adjustment

    def _calculate_cloud_impact(self, cloud_cover: float) -> float:
        """Calculate impact of cloud cover on solar irradiance."""
        if cloud_cover < 25:
            return ENVIRONMENT.get('clear_sky_factor', 1.0)
        elif cloud_cover < 50:
            return ENVIRONMENT.get('light_clouds_factor', 0.8)
        elif cloud_cover < 75:
            return ENVIRONMENT.get('medium_clouds_factor', 0.5)
        else:
            return ENVIRONMENT.get('heavy_clouds_factor', 0.2)

    def calculate_power(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate power generation for all waypoints.

        Args:
            weather_data: DataFrame containing weather data and waypoints.

        Returns:
            DataFrame with power calculations.
        """
        power_data = []

        for _, row in weather_data.iterrows():
            try:
                # Ensure timestamp is in datetime format
                timestamp = pd.to_datetime(row.get("timestamp"), errors='coerce')
                if pd.isnull(timestamp):
                    continue

                # Solar position
                solar_position = calculate_solar_position(
                    row.get("latitude", 0),
                    row.get("longitude", 0),
                    timestamp
                )

                # Weather factors with robust default values
                temperature = row.get("temperature", 25)
                cloud_cover = row.get("cloud_cover", 50)
                dni = row.get("DNI", 0)
                dhi = row.get("DHI", 0)
                ghi = row.get("GHI", 0)

                # Adjustments for temperature and cloud impact
                temperature_efficiency = self._adjust_efficiency_for_temperature(temperature)
                cloud_factor = self._calculate_cloud_impact(cloud_cover)

                # Incident irradiance calculation
                incident_irradiance = calculate_incident_irradiance(
                    {"normal_vector": [0, 0, 1]},
                    solar_position[0],
                    solar_position[1],
                    dni * cloud_factor,
                    dhi * cloud_factor,
                    ghi * cloud_factor
                )

                # Total power calculation
                total_power = (
                    incident_irradiance *
                    temperature_efficiency *
                    self.panel_area *
                    (1 - self.total_system_losses)
                )

                # Append calculated row data
                power_data.append({
                    "timestamp": timestamp,
                    "latitude": row.get("latitude", 0),
                    "longitude": row.get("longitude", 0),
                    "altitude": row.get("altitude", 0),
                    "total_power": max(0, total_power),
                    "temperature": temperature,
                    "cloud_cover": cloud_cover
                })

            except Exception as e:
                print(f"Error processing row: {e}")

        return pd.DataFrame(power_data)
# Example usage
if __name__ == "__main__":
    weather_file = "weather_data.csv"
    output_file = "power_output.csv"

    # Read weather data
    weather_data = pd.read_csv(weather_file)

    # Ensure timestamps are parsed correctly
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors='coerce')
    if weather_data["timestamp"].isnull().any():
        raise ValueError("Invalid timestamp format in input data.")

    # Instantiate calculator
    calculator = PowerCalculator()

    # Calculate power and save results
    power_df = calculator.calculate_power(weather_data)
    power_df.to_csv(output_file, index=False)
    print(f"Power data saved to {output_file}")
