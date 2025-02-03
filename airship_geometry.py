import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class AirshipParameters:
    """
    Configuration parameters for airship geometry.
    Modify these values to adjust the airship design.
    """
    length: float  # Length of airship in meters
    ld_ratio: float  # Length-to-diameter ratio
    profile_type: str = "NPL"  # Airship profile type: 'NPL', 'GNVR', etc.
    integration_points: int = 1000  # Points for numerical integration

class AirshipGeometry:
    def __init__(self, params: AirshipParameters):
        """
        Initialize airship geometry with configurable parameters.
        
        Args:
            params: AirshipParameters instance containing all configuration
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        self.params = params
        self._validate_parameters(params)
        self.diameter = params.length / params.ld_ratio
        self.radius = self.diameter / 2
        self._surface_area = None
        self._volume = None

    def _validate_parameters(self, params: AirshipParameters):
        """Validate airship parameters."""
        if params.length <= 0:
            raise ValueError("Length must be positive.")
        if params.ld_ratio < 2 or params.ld_ratio > 8:
            raise ValueError("L/D ratio should be between 2 and 8.")
        if params.profile_type not in ["NPL", "GNVR"]:
            raise ValueError("Invalid profile type. Choose 'NPL' or 'GNVR'.")

    def get_profile_equation(self, x: float) -> float:
        """
        Calculate the radius at a given point along the length using the selected profile type.
        Supports different airship profiles (NPL, GNVR).
        
        Args:
            x: Position along the airship length (0 to length)
        
        Returns:
            Radius at position x in meters
        """
        x_norm = (2 * x / self.params.length) - 1
        if self.params.profile_type == "NPL":
            return self.radius * (1 - x_norm**2)  # Parabolic profile
        elif self.params.profile_type == "GNVR":
            return self.radius * (1 - abs(x_norm)**1.5)  # GNVR-inspired profile
        else:
            raise ValueError("Invalid profile type.")

    def generate_panels(self, num_longitudinal: int, num_circumferential: int) -> List[Dict]:
        """
        Generate panels on the airship surface for geometry calculations.
        
        Args:
            num_longitudinal: Number of divisions along the length
            num_circumferential: Number of divisions around the circumference
        
        Returns:
            List of panel properties
        """
        if num_longitudinal < 10 or num_circumferential < 12:
            raise ValueError("Insufficient panels for accurate geometry.")
        
        panels = []
        x_positions = np.linspace(0, self.params.length, num_longitudinal)
        phi_angles = np.linspace(0, 2 * np.pi, num_circumferential, endpoint=False)
        
        for x in x_positions:
            r = self.get_profile_equation(x)
            for phi in phi_angles:
                y, z = r * np.cos(phi), r * np.sin(phi)
                panels.append({"x": x, "y": y, "z": z, "radius": r, "phi": phi})
        return panels

    def calculate_surface_area(self) -> float:
        """
        Compute the total surface area using numerical integration.
        
        Returns:
            Total surface area in square meters
        """
        if self._surface_area is not None:
            return self._surface_area
        
        x, weights = np.polynomial.legendre.leggauss(self.params.integration_points)
        x = self.params.length * (x + 1) / 2
        r = np.array([self.get_profile_equation(xi) for xi in x])
        dr_dx = np.gradient(r, x)
        integrand = 2 * np.pi * r * np.sqrt(1 + dr_dx**2)
        self._surface_area = np.sum(integrand * weights) * self.params.length / 2
        return self._surface_area

    def calculate_volume(self) -> float:
        """
        Compute the total volume using numerical integration.
        
        Returns:
            Total volume in cubic meters
        """
        if self._volume is not None:
            return self._volume
        
        x, weights = np.polynomial.legendre.leggauss(self.params.integration_points)
        x = self.params.length * (x + 1) / 2
        r = np.array([self.get_profile_equation(xi) for xi in x])
        self._volume = np.pi * np.sum(r**2 * weights) * self.params.length / 2
        return self._volume

def create_example_airship():
    """
    Create an airship with typical parameters for demonstration.
    """
    params = AirshipParameters(
        length=50.0,  # Length of 50 meters
        ld_ratio=4.0,  # Length/Diameter ratio of 4
        profile_type="NPL",  # NPL profile
        integration_points=1000
    )
    return AirshipGeometry(params)

if __name__ == "__main__":
    airship = create_example_airship()
    print(f"Surface Area: {airship.calculate_surface_area():.2f} m²")
    print(f"Volume: {airship.calculate_volume():.2f} m³")
