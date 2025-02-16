import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict
import logging
from functools import lru_cache
from datetime import timedelta
import traceback
import subprocess
import sys

# List of required dependencies
required_packages = [
    'tkinter',  # Example package, you can add more
    'numpy',    # Example package
    'matplotlib' # Example package
]


def install_missing_packages():
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"The following packages are missing: {', '.join(missing_packages)}")
        install = input("Do you want to install them? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("Packages installed successfully.")
        else:
            print("Please install the missing packages and try again.")
            sys.exit(1)


# Check for dependencies when the program starts
install_missing_packages()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class CelestialBody:
    radius: float  # Mm
    atmosphere_start: float  # Mm
    mass: float  # kg
    soi_radius: float  # Mm


class RelaySetup(TypedDict):
    num_relays: int
    h_low: float
    h_high: float
    recommended_altitude: float
    coverage_per_relay_rad: float
    coverage_per_relay_deg: float
    total_coverage_rad: float
    total_coverage_deg: float
    relay_distance: float
    orbital_period: float


# Celestial body data including mass and SOI
KSP_BODIES: Dict[str, CelestialBody] = {
    "Kerbin": CelestialBody(
        radius=0.6,
        atmosphere_start=0.07,
        mass=5.2915793e22,
        soi_radius=84559
    ),
    "Mun": CelestialBody(
        radius=0.2,
        atmosphere_start=0,
        mass=9.7599066e20,
        soi_radius=2.429559
    ),
    "Minmus": CelestialBody(
        radius=0.06,
        atmosphere_start=0,
        mass=2.6457580e19,
        soi_radius=2.247428
    ),
    "Duna": CelestialBody(
        radius=0.32,
        atmosphere_start=0.05,
        mass=4.5154270e21,
        soi_radius=47921.949
    ),
    "Eve": CelestialBody(
        radius=0.7,
        atmosphere_start=0.09,
        mass=1.2244127e23,
        soi_radius=85109.365
    ),
    "Ike": CelestialBody(
        radius=0.13,
        atmosphere_start=0,
        mass=2.7821940e20,
        soi_radius=1.049598
    ),
    "Gilly": CelestialBody(
        radius=0.013,
        atmosphere_start=0,
        mass=1.2420510e17,
        soi_radius=0.126164
    ),
    "Jool": CelestialBody(
        radius=6.0,
        atmosphere_start=0.2,
        mass=4.2332635e24,
        soi_radius=2455985.2
    ),
    "Laythe": CelestialBody(
        radius=0.5,
        atmosphere_start=0.055,
        mass=2.9397663e22,
        soi_radius=3.723646
    ),
    "Vall": CelestialBody(
        radius=0.3,
        atmosphere_start=0,
        mass=3.1088024e21,
        soi_radius=2.406401
    ),
    "Tylo": CelestialBody(
        radius=0.6,
        atmosphere_start=0,
        mass=4.2332635e22,
        soi_radius=10.856063
    ),
    "Bop": CelestialBody(
        radius=0.065,
        atmosphere_start=0,
        mass=3.7261090e19,
        soi_radius=1.221060
    ),
    "Pol": CelestialBody(
        radius=0.044,
        atmosphere_start=0,
        mass=1.0813500e19,
        soi_radius=0.934793
    ),
    "Dres": CelestialBody(
        radius=0.138,
        atmosphere_start=0,
        mass=3.2190937e20,
        soi_radius=32832.84
    ),
    "Eeloo": CelestialBody(
        radius=0.21,
        atmosphere_start=0,
        mass=1.1149358e21,
        soi_radius=11914.13
    ),
    "Moho": CelestialBody(
        radius=0.25,
        atmosphere_start=0,
        mass=2.5263314e21,
        soi_radius=9646.663
    )
}

G = 6.67430e-11  # Gravitational constant in m³/kg/s²


@dataclass
class OrbitalParameters:
    period: float  # seconds
    velocity: float  # m/s
    coverage_time: float  # seconds
    gap_time: float  # seconds

def calculate_orbital_parameters(body: CelestialBody, altitude: float) -> OrbitalParameters:
    """Calculate orbital parameters for a given altitude."""
    orbit_radius = (body.radius + altitude) * 1000000  # Convert Mm to m
    period = 2 * math.pi * math.sqrt(orbit_radius ** 3 / (G * body.mass))
    velocity = math.sqrt(G * body.mass / orbit_radius)
    coverage_time = period / 4  # Approximate time satellite is overhead
    gap_time = period / 4  # Approximate time between passes
    return OrbitalParameters(period, velocity, coverage_time, gap_time)


# Function to calculate resonant orbit parameters
def calculate_resonant_orbit(body, target_altitude_Mm, num_relays):
    """Calculate resonant orbit parameters for relay deployment."""
    # Convert values to meters
    body_radius_m = body.radius * 1e6  # Body radius in meters
    target_radius_m = body_radius_m + (target_altitude_Mm * 1e6)  # Target orbit radius in meters
    mu = 6.67430e-11 * body.mass  # Gravitational parameter

    # Target circular orbit period
    target_period = 2 * math.pi * math.sqrt(target_radius_m**3 / mu)

    # ----------------------------------------------------------------------------------
    # Apoapsis Release Strategy (drop relays at apoapsis, burn at periapsis)
    # Transfer period: (N+1)/N times target period
    transfer_period_apo = target_period * (num_relays + 1) / num_relays
    a_transfer_apo = ( (transfer_period_apo**2 * mu) / (4 * math.pi**2) ) ** (1/3)
    apo_release = {
        'periapsis': target_radius_m,  # Burn at periapsis (target altitude)
        'apoapsis': 2 * a_transfer_apo - target_radius_m  # Higher apoapsis
    }

    # ----------------------------------------------------------------------------------
    # Periapsis Release Strategy (drop relays at periapsis, burn at apoapsis)
    # Transfer period: N/(N+1) times target period
    transfer_period_peri = target_period * num_relays / (num_relays + 1)
    a_transfer_peri = ( (transfer_period_peri**2 * mu) / (4 * math.pi**2) ) ** (1/3)
    peri_release = {
        'apoapsis': target_radius_m,  # Burn at apoapsis (target altitude)
        'periapsis': 2 * a_transfer_peri - target_radius_m  # Lower periapsis
    }

    # ----------------------------------------------------------------------------------
    # Delta-V calculation
    def calculate_dv(burn_radius, other_apsis):  # Requires 2 arguments
        a_ellipse = (burn_radius + other_apsis) / 2
        v_initial = math.sqrt(mu * (2 / burn_radius - 1 / a_ellipse))
        v_final = math.sqrt(mu / burn_radius)
        return abs(v_final - v_initial)

    return {
        'resonance_ratio': (num_relays + 1, num_relays),
        'apoapsis_release': {
            'periapsis_altitude': (apo_release['periapsis'] - body_radius_m) / 1e6,
            'apoapsis_altitude': (apo_release['apoapsis'] - body_radius_m) / 1e6,
            # Pass BOTH arguments: burn radius (periapsis) and other apsis (apoapsis)
            'delta_v': calculate_dv(apo_release['periapsis'], apo_release['apoapsis'])
        },
        'periapsis_release': {
            'periapsis_altitude': (peri_release['periapsis'] - body_radius_m) / 1e6,
            'apoapsis_altitude': (peri_release['apoapsis'] - body_radius_m) / 1e6,
            # Pass BOTH arguments: burn radius (apoapsis) and other apsis (periapsis)
            'delta_v': calculate_dv(peri_release['apoapsis'], peri_release['periapsis'])
        }
    }
def coverage_angle_surface(r: float, h: float, R_surface: float) -> float:
    """
    Compute coverage angle for surface coverage.
    The chord length (on the surface) from the sub-satellite point to the horizon is:
       chord_h = 2 * r * sqrt(1 - (r/(r+h))^2)
    The effective chord is the minimum of R_surface and chord_h.
    The central angle (theta) is:
       theta = 2 * arcsin( effective_chord / (2*r) )
    """
    chord_h = 2 * r * math.sqrt(1 - (r / (r + h)) ** 2)
    effective_chord = min(R_surface, chord_h)
    theta = 2 * math.asin(effective_chord / (2 * r))
    return theta


def calc_relay_distance(r: float, h: float, num_relays: int) -> float:
    """Calculate distance between relays on an orbit of radius (r+h) with equal spacing."""
    orbital_radius = r + h
    return 2 * orbital_radius * math.sin(math.pi / num_relays)

def relay_los_check(body_radius: float, orbital_height: float, relay_range: float, num_relays: int) -> bool:
    """
    Checks if there is a clear line of sight between relays.
    Ensures that the relay range is sufficient to cover the distance between relays.
    """
    if num_relays < 2:  # Need at least 2 relays for LOS check
        return False

    # Angle between relays (in radians)
    angle_between_relays = 2 * math.pi / num_relays

    # Half of the angle between relays
    half_angle = angle_between_relays / 2

    # Minimum altitude for LOS (relays must be above this to see each other)
    min_altitude = body_radius * (1 / math.cos(half_angle) - 1)

    # Calculate the distance between relays at this orbital height
    orbital_radius = body_radius + orbital_height
    relay_distance = 2 * orbital_radius * math.sin(half_angle)

    # Check if the relay range is sufficient for the relay distance
    if relay_distance > relay_range:
        return False  # If relay range is smaller than relay distance, LOS check fails

    return orbital_height >= min_altitude

@lru_cache(maxsize=32)
def find_minimal_relay_setup(
        body_name: str,
        body_radius: float,
        surface_range: float,
        relay_range: float,
        max_relays: int = 20
) -> Optional[RelaySetup]:
    """Find minimal relay setup with caching."""
    if body_name not in KSP_BODIES:
        raise ValueError(f"Unknown celestial body: {body_name}")
    body = KSP_BODIES[body_name]
    h_ground = math.sqrt(surface_range ** 2 + body_radius ** 2) - body_radius
    best_result = None
    for N in range(3, max_relays + 1):
        h_LOS = body_radius * (1 / math.cos(math.pi / N) - 1)
        h_inter = relay_range / (2 * math.sin(math.pi / N)) - body_radius
        h_low = max(h_LOS, body.atmosphere_start)
        h_high = min(h_ground, h_inter)
        if h_low > h_high:
            continue
        h_candidate = (h_low + h_high) / 2
        orbital_radius = body_radius + h_candidate  # Calculate orbital radius
        theta = coverage_angle_surface(body_radius, h_candidate, surface_range)
        total_coverage = N * theta
        relay_distance = calc_relay_distance(body_radius, h_candidate, N)
        if relay_range >= orbital_radius and relay_distance <= relay_range:
            if total_coverage >= 2 * math.pi:
                orbital_params = calculate_orbital_parameters(body, h_candidate)
                result = {
                    'num_relays': N,
                    'h_low': h_low,
                    'h_high': h_high,
                    'recommended_altitude': h_candidate,
                    'coverage_per_relay_rad': theta,
                    'coverage_per_relay_deg': math.degrees(theta),
                    'total_coverage_rad': total_coverage,
                    'total_coverage_deg': math.degrees(total_coverage),
                    'relay_distance': relay_distance,
                    'orbital_period': orbital_params.period
                }
                if best_result is None or N < best_result['num_relays']:
                    best_result = result
    return best_result

class RelayNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KSP Relay Network Calculator")

        # Initialize control variables FIRST
        self.surface_range_var = tk.StringVar(value="1.5")
        self.relay_range_var = tk.StringVar(value="5.0")
        self.orbit_height_var = tk.StringVar()
        self.custom_relay_count_var = tk.StringVar()
        self.antenna_var = tk.StringVar(value="0")  # Initialize this here
        self.network_type_var = tk.StringVar(value="optimal")  # Initialize this here

        # Then create GUI elements
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # Create interface components
        self.create_input_frame()
        self.create_plot_frame()
        self.create_results_frame()

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create menu last
        self.create_menu()

        # Trigger initial update of entry states
        self.update_entry_states()

    def create_menu(self):
        menubar = tk.Menu(self.root)  # Create the menu bar

        helpmenu = tk.Menu(menubar, tearoff=0)  # Create the "Help" menu
        helpmenu.add_command(label="About", command=self.show_about_dialog)  # Add "About" item
        menubar.add_cascade(label="Help", menu=helpmenu)  # Add "Help" to the menu bar

        self.root.config(menu=menubar)  # Set the menu bar for the root window

    def show_about_dialog(self):
        about_text = ("\n"
                      "        KSP Relay Network Calculator\n"
                      "\n"
                      "        This program helps you design relay networks focused on achieving full equatorial surface coverage in RemoteTech mod\n"
                      "        Enter the celestial body, antenna configuration, and desired ranges.\n"
                      "        The program will find the minimal number of relays needed for full coverage, if setting a custom orbit, leaving the custom relay field blank will find the minimum relays needed at that orbit .\n"
                      "        ")  # Your program's description
        messagebox.showinfo("About", about_text)  # Show a message box with the info

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Input Parameters", padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Celestial Body Selection
        ttk.Label(input_frame, text="Celestial Body:").grid(row=0, column=0, sticky=tk.W)
        self.body_var = tk.StringVar(value="Kerbin")
        body_combo = ttk.Combobox(input_frame, textvariable=self.body_var)
        body_combo['values'] = list(KSP_BODIES.keys())
        body_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Network Type Selection
        ttk.Label(input_frame, text="Network Type:").grid(row=1, column=0, sticky=tk.W)
        self.network_type_var = tk.StringVar(value="optimal")
        ttk.Radiobutton(input_frame, text="Optimal Network", variable=self.network_type_var,
                        value="optimal").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(input_frame, text="Custom Network", variable=self.network_type_var,
                        value="custom").grid(row=2, column=1, sticky=tk.W)

        # Custom Network Parameters
        ttk.Label(input_frame, text="Custom Orbit Height (Mm):").grid(row=3, column=0, sticky=tk.W)
        self.orbit_height_entry = ttk.Entry(input_frame, textvariable=self.orbit_height_var, state='disabled')
        self.orbit_height_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Custom Relay Count:").grid(row=4, column=0, sticky=tk.W)
        self.custom_relay_entry = ttk.Entry(input_frame, textvariable=self.custom_relay_count_var, state='disabled')
        self.custom_relay_entry.grid(row=4, column=1, sticky=(tk.W, tk.E))

        # Spacer
        ttk.Separator(input_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)

        # Antenna Config
        ttk.Label(input_frame, text="Antenna Config:").grid(row=6, column=0, sticky=tk.W)
        self.antenna_var = tk.StringVar(value="0")
        configs = {
            "0": "2x 16-S (1.5 Mm)",
            "1": "16-S (1.5 Mm) + C-32 (5 Mm)",
            "2": "2x C-32 (5 Mm each)",
            "3": "Custom ranges"
        }

        for i, (key, value) in enumerate(configs.items()):
            ttk.Radiobutton(input_frame, text=value, variable=self.antenna_var,
                            value=key).grid(row=i + 7, column=1, sticky=tk.W, pady=2)

        # Antenna Range Entries
        self.surface_entry = ttk.Entry(input_frame, textvariable=self.surface_range_var, state='disabled')
        self.relay_entry = ttk.Entry(input_frame, textvariable=self.relay_range_var, state='disabled')

        ttk.Label(input_frame, text="Surface Range (Mm):").grid(row=11, column=0, sticky=tk.W)
        self.surface_entry.grid(row=11, column=1, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Relay Range (Mm):").grid(row=12, column=0, sticky=tk.W)
        self.relay_entry.grid(row=12, column=1, sticky=(tk.W, tk.E))

        # Calculate Button
        ttk.Button(input_frame, text="Calculate Network", command=self.calculate_network).grid(row=13, column=0,
                                                                                               columnspan=2, pady=10)

        # Trace setup
        self.antenna_var.trace('w', self.update_entry_states)
        self.network_type_var.trace('w', self.update_entry_states)
    def create_plot_frame(self):
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Network Visualization", padding="5")
        self.plot_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    def create_results_frame(self):
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        self.results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.results_text = tk.Text(self.results_frame, height=30, width=50)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text['yscrollcommand'] = scrollbar.set

    def update_entry_states(self, *args):
        # Handle antenna config entries
        if self.antenna_var.get() == "3":
            self.surface_entry['state'] = 'normal'
            self.relay_entry['state'] = 'normal'
        else:
            self.surface_entry['state'] = 'disabled'
            self.relay_entry['state'] = 'disabled'
            antenna_setting = self.antenna_var.get()
            # Explicit ordering for default case
            if antenna_setting == "0":  # 2x 16-S
                self.surface_range_var.set("1.5")
                self.relay_range_var.set("1.5")
            elif antenna_setting == "1":  # 16-S + C-32
                self.surface_range_var.set("1.5")
                self.relay_range_var.set("5.0")
            else:  # 2x C-32
                self.surface_range_var.set("5.0")
                self.relay_range_var.set("5.0")

        # Handle network type
        if self.network_type_var.get() == "custom":
            self.orbit_height_entry['state'] = 'normal'
            self.custom_relay_entry['state'] = 'normal'
        else:
            self.orbit_height_entry['state'] = 'disabled'
            self.custom_relay_entry['state'] = 'disabled'
    def calculate(self):
        try:
            body_name = self.body_var.get()
            surface_range = float(self.surface_range_var.get())
            relay_range = float(self.relay_range_var.get())
            body = KSP_BODIES[body_name]
            result = find_minimal_relay_setup(
                body_name,
                body.radius,
                surface_range,
                relay_range
            )
            if result:
                self.update_results(result, body)
                self.plot_network(result, body)
            else:
                messagebox.showerror("Error", "No valid configuration found")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            messagebox.showerror("Error", "An unexpected error occurred")

    def calculate_network(self):
        if self.network_type_var.get() == "optimal":
            self.calculate()  # Calls your existing calculate method
        else:
            self.custom_setup_gui()  # Calls your existing custom setup method

    def custom_setup_gui(self):
        try:
            custom_orbit = float(self.orbit_height_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid custom orbit height. Please enter a numeric value.")
            return

        body_name = self.body_var.get()
        body = KSP_BODIES[body_name]

        try:
            surface_range = float(self.surface_range_var.get())
            relay_range = float(self.relay_range_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid antenna range values.")
            return

        theta = coverage_angle_surface(body.radius, custom_orbit, surface_range)
        relay_count_str = self.custom_relay_count_var.get().strip()

        # If relay count is provided, validate it; otherwise, calculate minimum needed relays
        if relay_count_str:
            try:
                relay_count = int(relay_count_str)
                if relay_count < 3:
                    messagebox.showinfo("Info", "Relay count must be at least 3 for full coverage. Defaulting to 3.")
                    relay_count = 3
            except ValueError:
                messagebox.showerror("Error", "Invalid custom relay count. Please enter an integer.")
                return
        else:
            relay_count = math.ceil(2 * math.pi / theta)
            messagebox.showinfo("Info", f"Minimum relays needed for full coverage: {relay_count}")

        relay_distance = calc_relay_distance(body.radius, custom_orbit, relay_count)
        orbital_radius = body.radius + custom_orbit

        # Adjust relay count for LOS check
        while not relay_los_check(body.radius, custom_orbit, relay_range, relay_count):
            relay_count += 1
            print(f"Trying {relay_count} relays...")

        print(f"Final number of relays: {relay_count}")

        # Calculate orbital parameters
        orbital_params = calculate_orbital_parameters(body, custom_orbit)

        # Calculate resonant orbit data
        resonant_data = calculate_resonant_orbit(body, custom_orbit, relay_count)

        # Prepare and show the results
        results = [
            f"Custom Orbit Setup Results:",
            f"Orbital height: {custom_orbit:.2f} Mm",
            f"Coverage per relay: {math.degrees(theta):.2f}°",
            f"Total equatorial coverage: {(relay_count * math.degrees(theta) / 360) * 100:.2f}%",
            f"Number of relays: {relay_count}",
            f"Relay chord distance: {relay_distance:.2f} Mm",
            f"Orbital period: {timedelta(seconds=orbital_params.period)}",
            f"Orbital velocity: {orbital_params.velocity / 1000:.2f} km/s",

            "\n--- Resonant Orbit Options For Delivery Vessel---",
            f"Resonance Ratio: {resonant_data['resonance_ratio'][0]}:{resonant_data['resonance_ratio'][1]}",

            "\nOption 1: Release at Periapsis",
            f" - Periapsis Altitude: {resonant_data['apoapsis_release']['periapsis_altitude']:.6f} Mm",
            f" - Apoapsis Altitude: {resonant_data['apoapsis_release']['apoapsis_altitude']:.6f} Mm",
            f" - ΔV for circularization: {resonant_data['apoapsis_release']['delta_v']:.2f} m/s",

            "\nOption 2: Release at Apoapsis",
            f" - Periapsis Altitude: {resonant_data['periapsis_release']['periapsis_altitude']:.6f} Mm",
            f" - Apoapsis Altitude: {resonant_data['periapsis_release']['apoapsis_altitude']:.6f} Mm",
            f" - ΔV for circularization: {resonant_data['periapsis_release']['delta_v']:.2f} m/s"
        ]

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "\n".join(results))
        self.plot_network_custom(body, custom_orbit, relay_count, theta, relay_distance)
    def update_results(self, result, body):
        self.results_text.delete(1.0, tk.END)
        orbital_params = calculate_orbital_parameters(body, result['recommended_altitude'])
        resonant_data = calculate_resonant_orbit(body, result['recommended_altitude'], result['num_relays'])

        # Non-resonant results (core parameters)
        results = [
            f"Number of relays: {result['num_relays']}",
            f"Recommended orbital height: {result['recommended_altitude']:.2f} Mm",
            f"Feasible orbital window: {result['h_low']:.2f} Mm to {result['h_high']:.2f} Mm",
            f"Coverage per relay: {result['coverage_per_relay_deg']:.2f}°",
            f"Total coverage: {(result['total_coverage_deg'] / 360) * 100:.1f}%",
            f"Relay distance: {result['relay_distance']:.2f} Mm",
            f"Orbital period: {timedelta(seconds=orbital_params.period)} ",
            f"Orbital velocity: {orbital_params.velocity / 1000:.2f} km/s",
        ]

        # Add resonant orbit options
        results.extend([
            "\n--- Resonant Orbit Options For Delivery Vessel ---",
            f"Resonance Ratio: {resonant_data['resonance_ratio'][0]}:{resonant_data['resonance_ratio'][1]}",
            "\nOption 1: Release at Periapsis",
            f" - Periapsis Altitude: {resonant_data['apoapsis_release']['periapsis_altitude']:.6f} Mm",
            f" - Apoapsis Altitude: {resonant_data['apoapsis_release']['apoapsis_altitude']:.6f} Mm",
            f" - ΔV for circularization: {resonant_data['apoapsis_release']['delta_v']:.2f} m/s",
            "\nOption 2: Release at Apoapsis",
            f" - Periapsis Altitude: {resonant_data['periapsis_release']['periapsis_altitude']:.6f} Mm",
            f" - Apoapsis Altitude: {resonant_data['periapsis_release']['apoapsis_altitude']:.6f} Mm",
            f" - ΔV for circularization: {resonant_data['periapsis_release']['delta_v']:.2f} m/s"
        ])

        self.results_text.insert(tk.END, "\n".join(results))

    def plot_network(self, result: RelaySetup, body: CelestialBody):
        try:
            self.ax.clear()
            body_radius = body.radius
            orbital_radius = body_radius + result['recommended_altitude']
            num_relays = result['num_relays']
            coverage_angle_rad = result['coverage_per_relay_rad']
            circle = plt.Circle((0, 0), body_radius, color='blue', fill=True)
            self.ax.add_patch(circle)
            orbit = plt.Circle((0, 0), orbital_radius, color='gray', fill=False, linestyle='--')
            self.ax.add_patch(orbit)
            angles = np.linspace(0, 2 * np.pi, num_relays, endpoint=False)
            relay_x = orbital_radius * np.cos(angles)
            relay_y = orbital_radius * np.sin(angles)
            self.ax.scatter(relay_x, relay_y, color='red', s=50, zorder=3)
            for i in range(num_relays):
                relay_pos = np.array([relay_x[i], relay_y[i]])
                cone_start_angle = angles[i] - coverage_angle_rad / 2
                cone_end_angle = angles[i] + coverage_angle_rad / 2
                arc_angles = np.linspace(cone_start_angle, cone_end_angle, 50)
                arc_x = body_radius * np.cos(arc_angles)
                arc_y = body_radius * np.sin(arc_angles)
                cone_points = [(relay_pos[0], relay_pos[1])]
                cone_points.extend(zip(arc_x, arc_y))
                cone_x_vals, cone_y_vals = zip(*cone_points)
                self.ax.fill(cone_x_vals, cone_y_vals, color='orange', alpha=0.3)
            for i in range(num_relays):
                next_i = (i + 1) % num_relays
                self.ax.plot([relay_x[i], relay_x[next_i]],
                             [relay_y[i], relay_y[next_i]],
                             'g--', alpha=0.5)
                if i == 0:
                    midpoint_x = (relay_x[i] + relay_x[next_i]) / 2
                    midpoint_y = (relay_y[i] + relay_y[next_i]) / 2
                    self.ax.text(midpoint_x, midpoint_y, f"{result['relay_distance']:.2f} Mm", color="green",
                                 fontsize=10, ha="center", va="bottom")
            height_relay_index = 0
            height_x = relay_x[height_relay_index]
            height_y = relay_y[height_relay_index]
            body_edge_angle = np.arctan2(height_y, height_x)
            body_edge_x = body_radius * np.cos(body_edge_angle)
            body_edge_y = body_radius * np.sin(body_edge_angle)
            self.ax.plot([height_x, body_edge_x], [height_y, body_edge_y], color='purple', linestyle='-')
            mid_height_x = (height_x + body_edge_x) / 2
            mid_height_y = (height_y + body_edge_y) / 2
            self.ax.text(mid_height_x, mid_height_y + 0.02 * body_radius, f"{orbital_radius - body_radius:.2f} Mm",
                         color="purple", fontsize=10, ha="center", va="bottom")
            relay_artist = plt.Line2D([], [], color='red', marker='o', linestyle='', label='Relays')
            los_artist = plt.Line2D([], [], color='green', linestyle='--', label='Relay LOS')
            cone_artist = Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label='Surface Coverage')
            height_artist = plt.Line2D([], [], color='purple', linestyle='-', label='Orbital Height')
            self.ax.legend(handles=[relay_artist, los_artist, cone_artist, height_artist])
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title(f"Relay Network - {self.body_var.get()}")
            self.ax.set_xlabel("Distance (Mm)")
            self.ax.set_ylabel("Distance (Mm)")
            max_radius = orbital_radius * 1.2
            self.ax.set_xlim(-max_radius, max_radius)
            self.ax.set_ylim(-max_radius, max_radius)
            self.canvas.draw()
        except Exception as e:
            logging.error("Error occurred while plotting the network: %s", str(e))
            logging.error(traceback.format_exc())

    def plot_network_custom(self, body: CelestialBody, custom_orbit: float, relay_count: int, coverage_angle_rad: float,
                            relay_distance: float):
        try:
            self.ax.clear()
            body_radius = body.radius
            orbital_radius = body_radius + custom_orbit
            num_relays = relay_count
            circle = plt.Circle((0, 0), body_radius, color='blue', fill=True)
            self.ax.add_patch(circle)
            orbit = plt.Circle((0, 0), orbital_radius, color='gray', fill=False, linestyle='--')
            self.ax.add_patch(orbit)
            angles = np.linspace(0, 2 * np.pi, num_relays, endpoint=False)
            relay_x = orbital_radius * np.cos(angles)
            relay_y = orbital_radius * np.sin(angles)
            self.ax.scatter(relay_x, relay_y, color='red', s=50, zorder=3)
            for i in range(num_relays):
                relay_pos = np.array([relay_x[i], relay_y[i]])
                cone_start_angle = angles[i] - coverage_angle_rad / 2
                cone_end_angle = angles[i] + coverage_angle_rad / 2
                arc_angles = np.linspace(cone_start_angle, cone_end_angle, 50)
                arc_x = body_radius * np.cos(arc_angles)
                arc_y = body_radius * np.sin(arc_angles)
                cone_points = [(relay_pos[0], relay_pos[1])]
                cone_points.extend(zip(arc_x, arc_y))
                cone_x_vals, cone_y_vals = zip(*cone_points)
                self.ax.fill(cone_x_vals, cone_y_vals, color='orange', alpha=0.3)
            for i in range(num_relays):
                next_i = (i + 1) % num_relays
                self.ax.plot([relay_x[i], relay_x[next_i]], [relay_y[i], relay_y[next_i]], 'g--', alpha=0.5)
                if i == 0:
                    midpoint_x = (relay_x[i] + relay_x[next_i]) / 2
                    midpoint_y = (relay_y[i] + relay_y[next_i]) / 2
                    self.ax.text(midpoint_x, midpoint_y, f"{relay_distance:.2f} Mm", color="green", fontsize=10,
                                 ha="center", va="bottom")
            height_relay_index = 0
            height_x = relay_x[height_relay_index]
            height_y = relay_y[height_relay_index]
            body_edge_angle = np.arctan2(height_y, height_x)
            body_edge_x = body_radius * np.cos(body_edge_angle)
            body_edge_y = body_radius * np.sin(body_edge_angle)
            self.ax.plot([height_x, body_edge_x], [height_y, body_edge_y], color='purple', linestyle='-')
            mid_height_x = (height_x + body_edge_x) / 2
            mid_height_y = (height_y + body_edge_y) / 2
            self.ax.text(mid_height_x, mid_height_y + 0.02 * body_radius, f"{orbital_radius - body_radius:.2f} Mm",
                         color="purple", fontsize=10, ha="center", va="bottom")
            relay_artist = plt.Line2D([], [], color='red', marker='o', linestyle='', label='Relays')
            los_artist = plt.Line2D([], [], color='green', linestyle='--', label='Relay LOS')
            cone_artist = Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label='Surface Coverage')
            height_artist = plt.Line2D([], [], color='purple', linestyle='-', label='Orbital Height')
            self.ax.legend(handles=[relay_artist, los_artist, cone_artist, height_artist])
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title(f"Custom Relay Network - {self.body_var.get()}")
            self.ax.set_xlabel("Distance (Mm)")
            self.ax.set_ylabel("Distance (Mm)")
            max_radius = orbital_radius * 1.2
            self.ax.set_xlim(-max_radius, max_radius)
            self.ax.set_ylim(-max_radius, max_radius)
            self.canvas.draw()
        except Exception as e:
            logging.error("Error occurred while plotting custom network: %s", str(e))
            logging.error(traceback.format_exc())



if __name__ == "__main__":
    # Uncomment the following line to use the command-line version:
    # main()

    # For GUI version:
    root = tk.Tk()
    app = RelayNetworkGUI(root)
    root.mainloop()
