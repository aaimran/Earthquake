import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import os
import glob

app = dash.Dash(__name__)
server = app.server 

# Global cache for simulation data to avoid reloading the same file repeatedly.
simulation_cache = {}

# --- Identify Precomputed Simulation Files in datasets/ ---
datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
sim_files = glob.glob(os.path.join(datasets_dir, "simulation_tau_*.pkl"))

# Extract available computed τ and D_c values from filenames.
tau_set = set()
dc_set = set()
for file in sim_files:
    filename = os.path.basename(file)  # e.g., "simulation_tau_-1.00_dc_0.10.pkl"
    if filename.startswith("simulation_") and filename.endswith(".pkl"):
        core = filename[len("simulation_"):-len(".pkl")]  # "tau_-1.00_dc_0.10"
        parts = core.split("_")
        if len(parts) >= 4:
            try:
                tau_val = float(parts[1])
                dc_val = float(parts[3])
                tau_set.add(tau_val)
                dc_set.add(dc_val)
            except Exception as e:
                print("Error parsing simulation filename", filename, e)

tau_values = sorted(list(tau_set))
dc_values = sorted(list(dc_set))

# Set default parameters in case no values were found.
default_tau_min = tau_values[0] if tau_values else 81.24 - 0.36
default_tau_max = tau_values[-1] if tau_values else 81.24 + 0.36
default_tau_step = (tau_values[1] - tau_values[0]) if (tau_values and len(tau_values) > 1) else 0.072

app.layout = html.Div([
    html.H1("Earthquake Simulator"),
    dcc.Tabs([
        dcc.Tab(label='main', children=[
            # --- First Row: Simulation Parameter Sliders ---
            html.Div([
                html.Div([
                    html.Label("Initial Shear Stress [tau(e)]: 81.24 + e * 0.36 (MPa)"),
                    dcc.Slider(
                        id="tau-slider",
                        min=tau_values[0] if tau_values else default_tau_min,
                        max=tau_values[-1] if tau_values else default_tau_max,
                        step=default_tau_step,
                        value=tau_values[0] if tau_values else default_tau_min,
                        marks={round(val, 2): str(round(val, 2)) for val in tau_values} if tau_values 
                              else {default_tau_min: str(default_tau_min), default_tau_max: str(default_tau_max)}
                    )
                ], style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Critical Slip Distance [D_c] (m)"),
                    dcc.Slider(
                        id="dc-slider",
                        min=dc_values[0] if dc_values else 0.1,
                        max=dc_values[-1] if dc_values else 0.5,
                        step=0.1,
                        value=dc_values[0] if dc_values else 0.1,
                        marks={val: str(val) for val in dc_values} if dc_values 
                              else {0.1: "0.1", 0.5: "0.5"}
                    )
                ], style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px'}),
            
            # --- Second Row: Control Buttons and Time Slider in the same line ---
            html.Div([
                html.Div([
                    html.Button("Play", id="play-pause-button", n_clicks=0),
                    html.Button("Reset", id="reset-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Speed Up", id="speed-up-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Speed Down", id="speed-down-button", n_clicks=0, style={'marginLeft': '10px'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Time (s)"),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=10,
                        step=0.1,
                        value=0,
                        marks={0: "0", 10: "10"}
                    )
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'width': '60%'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            # Hidden Stores for simulation data and status.
            dcc.Store(id="simulation-data-store"),
            dcc.Store(id="simulation-status", data={"playing": False, "speed": 1.0}),
            
            # --- Bottom Panel: Graphs ---
            html.Div([
                # Left: Velocity and Stress Plots.
                html.Div([
                    dcc.Graph(id="velocity-graph", style={'width': '100%', 'height': '300px'}),
                    dcc.Graph(id="stress-graph", style={'width': '100%', 'height': '300px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                # Right: Time Series Plot.
                html.Div([
                    dcc.Graph(id="timeseries-graph", style={'width': '100%', 'height': '620px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            # Interval component for auto-advancing simulation time when playing.
            dcc.Interval(
                id="interval-component",
                interval=1000,  # 1 second (ms)
                n_intervals=0,
                disabled=True
            )
        ]),
        dcc.Tab(label='A', children=[
            html.Div("Placeholder content for Tab A.")
        ]),
        dcc.Tab(label='B', children=[
            html.Div("Placeholder content for Tab B.")
        ])
    ])
])

# Callback 1: Update simulation data based on τ and D_c slider values.
@app.callback(
    Output("simulation-data-store", "data"),
    [Input("tau-slider", "value"),
     Input("dc-slider", "value")]
)
def update_simulation_data(tau_val, dc_val):
    key = f"tau_{tau_val:.2f}_dc_{dc_val:.2f}"
    # Check if simulation data is already cached.
    if key in simulation_cache:
        return simulation_cache[key]
    
    filename = os.path.join(os.path.dirname(__file__), "datasets", f"simulation_{key}.pkl")
    try:
        with open(filename, "rb") as f:
            sim_data = pickle.load(f)
        # Convert numpy arrays to lists for JSON serialization.
        for field in ["DomainOutput_l", "DomainOutput_r", "y_l", "y_r"]:
            if field in sim_data and isinstance(sim_data[field], np.ndarray):
                sim_data[field] = sim_data[field].tolist()
        
        # Precompute and store velocity bounds.
        DomainOutput_l = np.array(sim_data["DomainOutput_l"])
        DomainOutput_r = np.array(sim_data["DomainOutput_r"])
        v_all = np.concatenate((DomainOutput_l[:, :, 0].flatten(), DomainOutput_r[:, :, 0].flatten()))
        sim_data["v_bounds"] = [float(np.min(v_all)), float(np.max(v_all))]
        
        # Precompute and store raw stress bounds (without the initial stress offset).
        s_all = np.concatenate((DomainOutput_l[:, :, 1].flatten(), DomainOutput_r[:, :, 1].flatten()))
        sim_data["s_bounds"] = [float(np.min(s_all)), float(np.max(s_all))]
        
        simulation_cache[key] = sim_data
        return sim_data
    except Exception as e:
        print(f"Error loading simulation data from {filename}: {e}")
        return None

# Callback 2: Update simulation status (play/pause, reset, speed up/down).
@app.callback(
    Output("simulation-status", "data"),
    [Input("play-pause-button", "n_clicks"),
     Input("reset-button", "n_clicks"),
     Input("speed-up-button", "n_clicks"),
     Input("speed-down-button", "n_clicks")],
    State("simulation-status", "data")
)
def update_simulation_status(play_clicks, reset_clicks, speed_up_clicks, speed_down_clicks, status):
    ctx = dash.callback_context
    if not ctx.triggered:
        return status
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "play-pause-button":
        status["playing"] = not status.get("playing", False)
    elif button_id == "reset-button":
        status["playing"] = False
        status["reset"] = True  # Flag to trigger a reset in the time slider callback.
    elif button_id == "speed-up-button":
        status["speed"] = status.get("speed", 1.0) + 0.5
    elif button_id == "speed-down-button":
        new_speed = status.get("speed", 1.0) - 0.5
        status["speed"] = new_speed if new_speed >= 0.1 else 0.1
    return status

# Callback 3: Enable/disable the Interval component based on play status.
@app.callback(
    Output("interval-component", "disabled"),
    Input("simulation-status", "data")
)
def toggle_interval(status):
    return not status.get("playing", False)

# Callback 4: Update the Time slider properties based on simulation data.
@app.callback(
    [Output("time-slider", "min"),
     Output("time-slider", "max"),
     Output("time-slider", "step"),
     Output("time-slider", "marks")],
    Input("simulation-data-store", "data")
)
def update_time_slider_properties(sim_data):
    if sim_data is None:
        return 0, 10, 0.1, {0: "0", 10: "10"}
    t_vector = np.array(sim_data["time_vector"])
    min_time = float(t_vector[0])
    max_time = float(t_vector[-1])
    step = sim_data["dt"]
    # Create 10 evenly spaced tick marks.
    marks = {round(t, 1): str(round(t, 1)) for t in np.linspace(min_time, max_time, num=10)}
    return min_time, max_time, step, marks

# Callback 5: Update the Time slider value.
@app.callback(
    Output("time-slider", "value"),
    [Input("interval-component", "n_intervals"),
     Input("reset-button", "n_clicks"),
     Input("speed-up-button", "n_clicks"),
     Input("speed-down-button", "n_clicks")],
    [State("time-slider", "value"),
     State("simulation-data-store", "data"),
     State("simulation-status", "data")]
)
def update_time_slider_value(n_intervals, reset_clicks, speed_up_clicks, speed_down_clicks,
                             current_time, sim_data, status):
    ctx = dash.callback_context
    if sim_data is None:
        return current_time
    t_vector = np.array(sim_data["time_vector"])
    min_time = float(t_vector[0])
    max_time = float(t_vector[-1])
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered == "reset-button":
        return min_time
    elif triggered == "interval-component":
        if status.get("playing", False):
            # Advance simulation time by the current speed factor.
            new_time = current_time + status.get("speed", 1.0)
            return new_time if new_time <= max_time else max_time
        else:
            return current_time
    # For speed up/down button clicks, do not change the current time immediately.
    return current_time

# Callback 6: Update graphs based on the Time slider value and tau-slider value.
@app.callback(
    [Output("velocity-graph", "figure"),
     Output("stress-graph", "figure"),
     Output("timeseries-graph", "figure")],
    [Input("time-slider", "value"),
     Input("simulation-data-store", "data"),
     Input("tau-slider", "value")]
)
def update_graphs(time_value, sim_data, tau_val):
    if not sim_data:
        return go.Figure(), go.Figure(), go.Figure()
    
    # Compute the initial stress using the given formula.
    initial_stress = 81.24 + tau_val * 0.36
    
    # Convert stored data back to numpy arrays.
    DomainOutput_l = np.array(sim_data["DomainOutput_l"])
    DomainOutput_r = np.array(sim_data["DomainOutput_r"])
    y_l = np.array(sim_data["y_l"])
    y_r = np.array(sim_data["y_r"])
    slip_vector = np.array(sim_data["slip_vector"])
    sliprate_vector = np.array(sim_data["sliprate_vector"])
    traction_vector = np.array(sim_data["traction_vector"])
    dt_sim = sim_data["dt"]
    
    # Convert time (s) to simulation index.
    current_index = int(round(time_value / dt_sim))
    if current_index >= DomainOutput_l.shape[1]:
        current_index = DomainOutput_l.shape[1] - 1
    displayed_time = current_index * dt_sim

    # Velocity Distribution Graph.
    fig_velocity = go.Figure()
    fig_velocity.add_trace(go.Scatter(
        x=y_l.flatten(),
        y=DomainOutput_l[:, current_index, 0],
        mode="lines+markers",
        name="Left Domain"
    ))
    fig_velocity.add_trace(go.Scatter(
        x=y_r.flatten(),
        y=DomainOutput_r[:, current_index, 0],
        mode="lines+markers",
        name="Right Domain"
    ))
    fig_velocity.update_layout(
        title=f"Velocity Distribution at t = {displayed_time:.2f} s",
        xaxis_title="x [km]",
        yaxis_title="Velocity [m/s]"
        # Auto-scaling enabled by omitting yaxis_range.
    )
    
    # Stress Distribution Graph.
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(
        x=y_l.flatten(),
        y=DomainOutput_l[:, current_index, 1] + initial_stress,
        mode="lines+markers",
        name="Left Domain"
    ))
    fig_stress.add_trace(go.Scatter(
        x=y_r.flatten(),
        y=DomainOutput_r[:, current_index, 1] + initial_stress,
        mode="lines+markers",
        name="Right Domain"
    ))
    fig_stress.update_layout(
        title=f"Stress Distribution at t = {displayed_time:.2f} s",
        xaxis_title="x [km]",
        yaxis_title="Stress [MPa]"
        # Auto-scaling enabled by omitting yaxis_range.
    )
    
    # Time Series Graph (Slip, Slip Rate, and Traction).
    t_data = np.linspace(0, displayed_time, current_index + 1)
    downsample_factor = 10
    if len(t_data) > downsample_factor:
        t_data_ds = t_data[::downsample_factor]
        slip_vector_ds = slip_vector[:current_index+1][::downsample_factor]
        sliprate_vector_ds = sliprate_vector[:current_index+1][::downsample_factor]
        traction_vector_ds = traction_vector[:current_index+1][::downsample_factor]
    else:
        t_data_ds = t_data
        slip_vector_ds = slip_vector[:current_index+1]
        sliprate_vector_ds = sliprate_vector[:current_index+1]
        traction_vector_ds = traction_vector[:current_index+1]
    
    fig_timeseries = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Slip", "Slip Rate", "Traction"),
        shared_xaxes=True
    )
    fig_timeseries.add_trace(go.Scatter(
        x=t_data_ds,
        y=slip_vector_ds,
        mode="lines+markers",
        name="Slip"
    ), row=1, col=1)
    fig_timeseries.add_trace(go.Scatter(
        x=t_data_ds,
        y=sliprate_vector_ds,
        mode="lines+markers",
        name="Slip Rate"
    ), row=2, col=1)
    fig_timeseries.add_trace(go.Scatter(
        x=t_data_ds,
        y=traction_vector_ds,
        mode="lines+markers",
        name="Traction"
    ), row=3, col=1)
    fig_timeseries.update_layout(
        title="Time Series: Slip, Slip Rate, and Traction",
        xaxis3_title="Time [s]",
        height=800
    )
    # Auto-scaling is handled by Plotly as no explicit axis ranges are set.
    
    return fig_velocity, fig_stress, fig_timeseries

if __name__ == '__main__':
    # For production on Windows, disable debug mode and the reloader.
    app.run_server(debug=False, use_reloader=False)