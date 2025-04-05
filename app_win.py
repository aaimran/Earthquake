import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import os
import glob
from pathlib import Path

# Use Path for cross-platform compatibility
app = dash.Dash(__name__)
server = app.server

# --- Identify Precomputed Simulation Files in datasets/ ---
datasets_dir = Path(__file__).parent / "datasets"
sim_files = glob.glob(str(datasets_dir / "simulation_tau_*.pkl"))

# Extract available computed τ and D_c values from filenames
tau_set = set()
dc_set = set()
for file in sim_files:
    filename = Path(file).name
    if filename.startswith("simulation_") and filename.endswith(".pkl"):
        core = filename[len("simulation_"):-len(".pkl")]
        parts = core.split("_")
        if len(parts) >= 4:
            try:
                tau_val = float(parts[1])
                dc_val = float(parts[3])
                tau_set.add(tau_val)
                dc_set.add(dc_val)
            except ValueError as e:
                print(f"Error parsing simulation filename {filename}: {e}")

tau_values = sorted(tau_set)
dc_values = sorted(dc_set)

# Set default parameters
default_tau_min = tau_values[0] if tau_values else 81.24 - 0.36
default_tau_max = tau_values[-1] if tau_values else 81.24 + 0.36
default_tau_step = (tau_values[1] - tau_values[0]) if (tau_values and len(tau_values) > 1) else 0.072

# Optimized layout with lazy loading
app.layout = html.Div([
    html.H1("Earthquake Simulator"),
    dcc.Tabs([
        dcc.Tab(label='Main', children=[
            html.Div([
                html.Div([
                    html.Label("Initial Shear Stress [tau(e)]: 81.24 + e * 0.36 (MPa)"),
                    dcc.Slider(id="tau-slider", min=default_tau_min, max=default_tau_max, 
                             step=default_tau_step, value=default_tau_min,
                             marks={round(val, 2): str(round(val, 2)) for val in tau_values} or 
                                   {default_tau_min: str(default_tau_min), default_tau_max: str(default_tau_max)})
                ], style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Critical Slip Distance [D_c] (m)"),
                    dcc.Slider(id="dc-slider", min=dc_values[0] if dc_values else 0.1,
                             max=dc_values[-1] if dc_values else 0.5, step=0.1,
                             value=dc_values[0] if dc_values else 0.1,
                             marks={val: str(val) for val in dc_values} or {0.1: "0.1", 0.5: "0.5"})
                ], style={'width': '300px', 'display': 'inline-block'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px'}),
            
            html.Div([
                html.Div([
                    html.Button("Play", id="play-pause-button", n_clicks=0),
                    html.Button("Reset", id="reset-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Speed Up", id="speed-up-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Speed Down", id="speed-down-button", n_clicks=0, style={'marginLeft': '10px'})
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Time (s)"),
                    dcc.Slider(id="time-slider", min=0, max=10, step=0.1, value=0,
                             marks={0: "0", 10: "10"})
                ], style={'display': 'inline-block', 'width': '60%'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            dcc.Store(id="simulation-data-store"),
            dcc.Store(id="simulation-status", data={"playing": False, "speed": 1.0}),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="velocity-graph", style={'height': '300px'}),
                    dcc.Graph(id="stress-graph", style={'height': '300px'})
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id="timeseries-graph", style={'height': '620px'})
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True)
        ]),
        dcc.Tab(label='A', children=[html.Div("Placeholder content for Tab A.")]),
        dcc.Tab(label='B', children=[html.Div("Placeholder content for Tab B.")])
    ])
])

# Callback 1: Load simulation data efficiently
@app.callback(
    Output("simulation-data-store", "data"),
    [Input("tau-slider", "value"), Input("dc-slider", "value")]
)
def update_simulation_data(tau_val, dc_val):
    key = f"tau_{tau_val:.2f}_dc_{dc_val:.2f}"
    filename = datasets_dir / f"simulation_{key}.pkl"
    try:
        with open(filename, "rb") as f:
            sim_data = pickle.load(f)
        # Minimize memory usage by converting only necessary fields
        for field in ["DomainOutput_l", "DomainOutput_r", "y_l", "y_r"]:
            if field in sim_data and isinstance(sim_data[field], np.ndarray):
                sim_data[field] = sim_data[field].tolist()
        return sim_data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Callback 2: Update simulation status
@app.callback(
    Output("simulation-status", "data"),
    [Input("play-pause-button", "n_clicks"), Input("reset-button", "n_clicks"),
     Input("speed-up-button", "n_clicks"), Input("speed-down-button", "n_clicks")],
    State("simulation-status", "data")
)
def update_simulation_status(play_clicks, reset_clicks, speed_up_clicks, speed_down_clicks, status):
    ctx = dash.callback_context
    if not ctx.triggered:
        return status
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    status = status.copy()  # Avoid mutating state directly
    if button_id == "play-pause-button":
        status["playing"] = not status.get("playing", False)
    elif button_id == "reset-button":
        status["playing"] = False
        status["reset"] = True
    elif button_id == "speed-up-button":
        status["speed"] = min(status.get("speed", 1.0) + 0.5, 5.0)  # Cap max speed
    elif button_id == "speed-down-button":
        status["speed"] = max(status.get("speed", 1.0) - 0.5, 0.1)  # Cap min speed
    return status

# Callback 3: Toggle interval
@app.callback(
    Output("interval-component", "disabled"),
    Input("simulation-status", "data")
)
def toggle_interval(status):
    return not status.get("playing", False)

# Callback 4: Update time slider properties
@app.callback(
    [Output("time-slider", "min"), Output("time-slider", "max"),
     Output("time-slider", "step"), Output("time-slider", "marks")],
    Input("simulation-data-store", "data")
)
def update_time_slider_properties(sim_data):
    if not sim_data:
        return 0, 10, 0.1, {0: "0", 10: "10"}
    t_vector = np.array(sim_data["time_vector"])
    min_time, max_time = float(t_vector[0]), float(t_vector[-1])
    step = sim_data["dt"]
    marks = {int(t): str(int(t)) for t in np.linspace(min_time, max_time, num=5)}  # Fewer marks for performance
    return min_time, max_time, step, marks

# Callback 5: Update time slider value
@app.callback(
    Output("time-slider", "value"),
    [Input("interval-component", "n_intervals"), Input("reset-button", "n_clicks"),
     Input("speed-up-button", "n_clicks"), Input("speed-down-button", "n_clicks")],
    [State("time-slider", "value"), State("simulation-data-store", "data"),
     State("simulation-status", "data")]
)
def update_time_slider_value(n_intervals, reset_clicks, speed_up_clicks, speed_down_clicks,
                            current_time, sim_data, status):
    if not sim_data:
        return current_time
    t_vector = np.array(sim_data["time_vector"])
    min_time, max_time = float(t_vector[0]), float(t_vector[-1])
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered == "reset-button":
        return min_time
    elif triggered == "interval-component" and status.get("playing", False):
        new_time = current_time + status.get("speed", 1.0) * sim_data["dt"]
        return min(new_time, max_time)
    return current_time

# Callback 6: Update graphs with optimized rendering
@app.callback(
    [Output("velocity-graph", "figure"), Output("stress-graph", "figure"),
     Output("timeseries-graph", "figure")],
    [Input("time-slider", "value"), Input("simulation-data-store", "data"),
     Input("tau-slider", "value")]
)
def update_graphs(time_value, sim_data, tau_val):
    if not sim_data:
        return go.Figure(), go.Figure(), go.Figure()

    initial_stress = 81.24 + tau_val * 0.36
    DomainOutput_l = np.array(sim_data["DomainOutput_l"])
    DomainOutput_r = np.array(sim_data["DomainOutput_r"])
    y_l, y_r = np.array(sim_data["y_l"]), np.array(sim_data["y_r"])
    slip_vector = np.array(sim_data["slip_vector"])
    sliprate_vector = np.array(sim_data["sliprate_vector"])
    traction_vector = np.array(sim_data["traction_vector"])
    dt_sim = sim_data["dt"]

    current_index = min(int(round(time_value / dt_sim)), DomainOutput_l.shape[1] - 1)
    displayed_time = current_index * dt_sim

    # Precompute bounds once
    v_min, v_max = np.min([DomainOutput_l[:, :, 0], DomainOutput_r[:, :, 0]]), np.max([DomainOutput_l[:, :, 0], DomainOutput_r[:, :, 0]])
    s_min, s_max = np.min([DomainOutput_l[:, :, 1] + initial_stress, DomainOutput_r[:, :, 1] + initial_stress]), np.max([DomainOutput_l[:, :, 1] + initial_stress, DomainOutput_r[:, :, 1] + initial_stress])

    # Downsample time series data
    t_data = np.linspace(0, displayed_time, current_index + 1)
    downsample_factor = max(1, len(t_data) // 100)  # Limit to ~100 points
    t_data_ds = t_data[::downsample_factor]
    slip_ds = slip_vector[:current_index + 1:downsample_factor]
    sliprate_ds = sliprate_vector[:current_index + 1:downsample_factor]
    traction_ds = traction_vector[:current_index + 1:downsample_factor]

    # Velocity Graph
    fig_velocity = go.Figure()
    fig_velocity.add_trace(go.Scatter(x=y_l, y=DomainOutput_l[:, current_index, 0], mode="lines", name="Left"))
    fig_velocity.add_trace(go.Scatter(x=y_r, y=DomainOutput_r[:, current_index, 0], mode="lines", name="Right"))
    fig_velocity.update_layout(title=f"Velocity at t = {displayed_time:.2f} s", xaxis_title="x [km]", yaxis_title="Velocity [m/s]", yaxis_range=[v_min, v_max])

    # Stress Graph
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(x=y_l, y=DomainOutput_l[:, current_index, 1] + initial_stress, mode="lines", name="Left"))
    fig_stress.add_trace(go.Scatter(x=y_r, y=DomainOutput_r[:, current_index, 1] + initial_stress, mode="lines", name="Right"))
    fig_stress.update_layout(title=f"Stress at t = {displayed_time:.2f} s", xaxis_title="x [km]", yaxis_title="Stress [MPa]", yaxis_range=[s_min, s_max])

    # Time Series Graph
    fig_timeseries = make_subplots(rows=3, cols=1, subplot_titles=("Slip", "Slip Rate", "Traction"), shared_xaxes=True)
    fig_timeseries.add_trace(go.Scatter(x=t_data_ds, y=slip_ds, mode="lines", name="Slip"), row=1, col=1)
    fig_timeseries.add_trace(go.Scatter(x=t_data_ds, y=sliprate_ds, mode="lines", name="Slip Rate"), row=2, col=1)
    fig_timeseries.add_trace(go.Scatter(x=t_data_ds, y=traction_ds, mode="lines", name="Traction"), row=3, col=1)
    fig_timeseries.update_layout(title="Time Series", xaxis3_title="Time [s]", height=600)

    return fig_velocity, fig_stress, fig_timeseries

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050, threaded=True)