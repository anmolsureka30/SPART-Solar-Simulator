import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def plot_power_output(power_output_file, output_dir):
    """
    Plots the total power output across the waypoints.

    :param power_output_file: Path to the CSV file with power output data.
    :param output_dir: Directory to save the plots.
    """
    try:
        data = pd.read_csv(power_output_file)
        if data.empty:
            raise ValueError("Power output file is empty.")

        # Ensure timestamp column is parsed as datetime
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Interactive plot using Plotly
        fig = px.line(
            data,
            x="timestamp",
            y="total_power",
            title="Total Solar Power Output Across the Flight Path",
            labels={"timestamp": "Timestamp", "total_power": "Power Output (Watts)"},
            markers=True
        )
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Power Output (Watts)",
            template="plotly_white",
            xaxis_tickangle=-45
        )

        # Save the interactive plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "power_output_plot.html")
        fig.write_html(output_path)
        print(f"Interactive power output plot saved to {output_path}")

    except Exception as e:
        print(f"Error in plot_power_output: {e}")

def plot_flight_path(waypoints_file, output_dir):
    """
    Visualizes the flight path on a world map.

    :param waypoints_file: Path to the CSV file with waypoints.
    :param output_dir: Directory to save the plots.
    """
    try:
        data = pd.read_csv(waypoints_file)
        if data.empty:
            raise ValueError("Waypoints file is empty.")

        # Check for required columns
        if not {"Longitude", "Latitude"}.issubset(data.columns):
            raise ValueError("Waypoints file must contain 'Longitude' and 'Latitude' columns.")

        # Static map using Cartopy
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_title("Flight Path of the Solar-Powered Airship", fontsize=14)

        # Add features to the map
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.LAKES, facecolor='blue')
        ax.add_feature(cfeature.RIVERS, edgecolor='blue')

        # Plot flight path
        ax.plot(
            data["Longitude"],
            data["Latitude"],
            marker="o",
            color="red",
            label="Flight Path",
            transform=ccrs.PlateCarree()
        )
        ax.legend(loc="upper left")

        # Save the static plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "flight_path_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Flight path plot saved to {output_path}")

    except Exception as e:
        print(f"Error in plot_flight_path: {e}")

def visualize_weather_data(weather_data_file, output_dir):
    """
    Visualizes weather data along the route (cloud cover, temperature, etc.).

    :param weather_data_file: Path to the CSV file with weather data.
    :param output_dir: Directory to save the plots.
    """
    try:
        data = pd.read_csv(weather_data_file)
        if data.empty:
            raise ValueError("Weather data file is empty.")

        # Ensure timestamp column is parsed as datetime
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Interactive weather data plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["timestamp"],
            y=data["cloud_cover"],
            mode="lines+markers",
            name="Cloud Cover (%)",
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data["timestamp"],
            y=data["temperature"],
            mode="lines+markers",
            name="Temperature (°C)",
            line=dict(color='orange')
        ))

        fig.update_layout(
            title="Weather Data Across the Flight Path",
            xaxis_title="Timestamp",
            yaxis_title="Value",
            template="plotly_white",
            xaxis_tickangle=-45
        )

        # Save the interactive plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "weather_data_plot.html")
        fig.write_html(output_path)
        print(f"Interactive weather data plot saved to {output_path}")

    except Exception as e:
        print(f"Error in visualize_weather_data: {e}")

def create_summary_table(flight_data_file, output_dir):
    """
    Generate a structured table showing solar power generation and other details.

    :param flight_data_file: Path to the CSV file with flight data.
    :param output_dir: Directory to save the table and plot.
    """
    try:
        # Read flight data
        data = pd.read_csv(flight_data_file)
        if data.empty:
            raise ValueError("Flight data file is empty.")

        # Check for required columns
        required_columns = {"timestamp", "latitude", "longitude", "solar_irradiance", "solar_power"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

        # Convert timestamp to datetime
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Generate summary table
        summary_table = data[["timestamp", "latitude", "longitude", "solar_irradiance", "solar_power"]]
        os.makedirs(output_dir, exist_ok=True)
        table_path = os.path.join(output_dir, "flight_summary_table.csv")
        summary_table.to_csv(table_path, index=False)
        print(f"Summary table saved to {table_path}")

        # Create a visualization of power generation over time
        fig = px.line(
            summary_table,
            x="timestamp",
            y="solar_power",
            title="Total Solar Power Generation Over Time",
            labels={"timestamp": "Timestamp", "solar_power": "Power Output (Watts)"},
            markers=True
        )
        fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
        plot_path = os.path.join(output_dir, "solar_power_plot.html")
        fig.write_html(plot_path)
        print(f"Solar power generation plot saved to {plot_path}")

    except Exception as e:
        print(f"Error in create_summary_table: {e}")
