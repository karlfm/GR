import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotter

def plot_results(file_name):
    # Configure a slick, professional plotting style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')  # Fallback style

    # Fine-tune the aesthetics - significantly increased sizes
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,              # Base font size
        'axes.titlesize': 20,         # Much bigger titles
        'axes.labelsize': 20,         # Bigger axis labels
        # 'axes.titleweight': 'bold',   # Bold titles
        'xtick.labelsize': 14,        # Bigger x-axis ticks
        'ytick.labelsize': 14,        # Bigger y-axis ticks
        'legend.fontsize': 16,        # Bigger legend text
        'legend.title_fontsize': 16,  # Bigger legend title
        'legend.frameon': True,
        'legend.framealpha': 0.95,    # Opaque background for readability
        'legend.fancybox': True,      # Rounded corners
        'legend.loc': 'best',         # Automatically choose the best location
        'lines.linewidth': 3,         # Thicker lines
        'grid.alpha': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    data_loaded = json.loads(Path(file_name).read_text())
    R_range = np.array(data_loaded["R_range"])
    dt = data_loaded["dt"]
    number_of_lines = data_loaded["number_of_lines"]
    stress_set_point = data_loaded["stress_set_point"]
    plot_data = data_loaded["plot_data_1d"]
    power_data = data_loaded["power_data"]

    plotter_instance = plotter.ComparisonPlotter(R_range, number_of_lines, model_name="LT")

    # Plot spatial data
    plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data["radial_stress"])
    plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data["hoop_stress"], set_point=stress_set_point)#set_point=stress_set_point)
    plotter_instance.plot_spatial_panel((0, 1), "Radial Stretch", "Stretch", plot_data["radial_strain"])
    plotter_instance.plot_spatial_panel((1, 1), "Hoop Stretch", "Stretch", plot_data["hoop_strain"])
    plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data["radial_growth"])
    plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data["hoop_growth"], plot_data["Homeostasis"])

    # Finalize and save
    plotter_instance.finalize_and_save("ODE_LT2_results.png")

if __name__ == "__main__":
    plot_results("LT2_ODE_data.json")