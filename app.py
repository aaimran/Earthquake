import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import os
import glob

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

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Earthquake Simulator"),
    dcc.Tabs([
        dcc.Tab(label='main', children=[
            # --- Top Panel: Control Buttons and Sliders in one row ---
            html.Div([
                # Control Buttons
                html.Div([
                    html.Button("Start", id="start-button", n_clicks=0),
                    html.Button("Pause", id="pause-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Stop", id="stop-button", n_clicks=0, style={'marginLeft': '10px'}),
                    html.Button("Reset", id="reset-button", n_clicks=0, style={'marginLeft': '10px'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}),
                
                # Sliders: τ slider, D_c slider, and Animation Speed slider
                html.Div([
                    html.Div([
                        html.Label("τ(e) : 81.24 + e×0.36"),
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
                        html.Label("D_c (m)"),
                        dcc.Slider(
                            id="dc-slider",
                            min=dc_values[0] if dc_values else 0.1,
                            max=dc_values[-1] if dc_values else 0.5,
                            step=0.1,
                            value=dc_values[0] if dc_values else 0.1,
                            marks={val: str(val) for val in dc_values} if dc_values 
                                  else {0.1: "0.1", 0.5: "0.5"}
                        )
                    ], style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Animation Speed (s)"),
                        dcc.Slider(
                            id="anim-speed-slider",
                            min=0.1,
                            max=2,
                            step=None,  # Allow only discrete values
                            value=1.0,  # Set default to 1 second per update
                            marks={
                                0.1: "0.1",
                                0.5: "0.5",
                                1: "1",
                                1.5: "1.5",
                                2: "2"
                            }
                        )
                    ], style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'padding': '10px'}),
            
            # Hidden Stores for simulation data and status.
            dcc.Store(id="simulation-data-store"),
            dcc.Store(id="simulation-status", data={"running": False, "stopped": False, "reset": False}),
            dcc.Store(id="simulation-index", data=0),
            
            # --- Bottom Panel: Divided into Left and Right Panels ---
            html.Div([
                # Bottom Left: Velocity and Stress Plots
                html.Div([
                    dcc.Graph(id="velocity-graph", style={'width': '100%', 'height': '300px'}),
                    dcc.Graph(id="stress-graph", style={'width': '100%', 'height': '300px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Bottom Right: Time Series Plot
                html.Div([
                    dcc.Graph(id="timeseries-graph", style={'width': '100%', 'height': '620px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            # Interval component for simulation playback.
            dcc.Interval(
                id="interval-component",
                interval=1.0 * 1000,  # default interval 1 second (ms)
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

# Callback 1: Update simulation data based on slider values by loading from the datasets/ folder.
@app.callback(
    Output("simulation-data-store", "data"),
    [Input("tau-slider", "value"),
     Input("dc-slider", "value")]
)
def update_simulation_data(tau_val, dc_val):
    key = f"tau_{tau_val:.2f}_dc_{dc_val:.2f}"
    filename = os.path.join(os.path.dirname(__file__), "datasets", f"simulation_{key}.pkl")
    try:
        with open(filename, "rb") as f:
            sim_data = pickle.load(f)
        # Convert numpy arrays to lists for JSON serialization, if necessary.
        fields_to_convert = ["DomainOutput_l", "DomainOutput_r", "y_l", "y_r"]
        for field in fields_to_convert:
            if field in sim_data and isinstance(sim_data[field], np.ndarray):
                sim_data[field] = sim_data[field].tolist()
        return sim_data
    except Exception as e:
        print(f"Error loading simulation data from {filename}: {e}")
        return None

# Callback 2: Update simulation status based on control buttons.
@app.callback(
    Output("simulation-status", "data"),
    [Input("start-button", "n_clicks"),
     Input("pause-button", "n_clicks"),
     Input("stop-button", "n_clicks"),
     Input("reset-button", "n_clicks")],
    State("simulation-status", "data")
)
def update_simulation_status(start_clicks, pause_clicks, stop_clicks, reset_clicks, status):
    ctx = dash.callback_context
    if not ctx.triggered:
        return status
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-button":
        status["running"] = True
        status["stopped"] = False
        status["reset"] = False
    elif button_id == "pause-button":
        status["running"] = False
    elif button_id in ["stop-button", "reset-button"]:
        status["running"] = False
        status["stopped"] = True
        status["reset"] = True
    return status

# Callback 3: Enable/disable the Interval based on simulation status.
@app.callback(
    Output("interval-component", "disabled"),
    Input("simulation-status", "data")
)
def toggle_interval(status):
    return not status.get("running", False)

# Callback 4: Update the simulation index.
# Here we update the index by a jump equal to 1 second of simulation time.
@app.callback(
    Output("simulation-index", "data"),
    [Input("interval-component", "n_intervals"),
     Input("simulation-status", "data"),
     Input("simulation-data-store", "data")],
    State("simulation-index", "data")
)
def update_sim_index(n_intervals, sim_status, sim_data, current_index):
    if sim_status.get("reset", False):
        return 0
    if not sim_status.get("running", False):
        return current_index
    # Calculate jump size (number of indices to advance) so that the simulation advances 1 second.
    if sim_data is not None and "dt" in sim_data:
        dt_sim = sim_data["dt"]
        jump = int(round(1.0 / dt_sim))
    else:
        jump = 1
    new_index = current_index + jump
    if sim_data is not None and "DomainOutput_l" in sim_data:
        max_index = len(sim_data["DomainOutput_l"][0]) - 1  # time dimension
        if new_index > max_index:
            new_index = max_index
    return new_index

# Callback 5: Update graphs based on simulation index and data.
@app.callback(
    [Output("velocity-graph", "figure"),
     Output("stress-graph", "figure"),
     Output("timeseries-graph", "figure")],
    [Input("simulation-index", "data"),
     Input("simulation-data-store", "data")]
)
def update_graphs(sim_index, sim_data):
    if not sim_data:
        return go.Figure(), go.Figure(), go.Figure()
    
    # Convert stored data back to numpy arrays.
    DomainOutput_l = np.array(sim_data["DomainOutput_l"])
    DomainOutput_r = np.array(sim_data["DomainOutput_r"])
    y_l = np.array(sim_data["y_l"])
    y_r = np.array(sim_data["y_r"])
    slip_vector = np.array(sim_data["slip_vector"])
    sliprate_vector = np.array(sim_data["sliprate_vector"])
    traction_vector = np.array(sim_data["traction_vector"])
    time_vector = np.array(sim_data["time_vector"])
    dt_sim = sim_data["dt"]
    
    # Determine global bounds for the plots using the full data set.
    # For Velocity (first component of DomainOutput arrays)
    v_all = np.concatenate((DomainOutput_l[:, :, 0].flatten(), DomainOutput_r[:, :, 0].flatten()))
    v_min, v_max = np.min(v_all), np.max(v_all)
    
    # For Stress (second component of DomainOutput arrays)
    s_all = np.concatenate((DomainOutput_l[:, :, 1].flatten(), DomainOutput_r[:, :, 1].flatten()))
    s_min, s_max = np.min(s_all), np.max(s_all)
    
    # For Time Series plots:
    slip_min, slip_max = np.min(slip_vector), np.max(slip_vector)
    sliprate_min, sliprate_max = np.min(sliprate_vector), np.max(sliprate_vector)
    traction_min, traction_max = np.min(traction_vector), np.max(traction_vector)
    
    # Limit the current index to available data.
    current_index = sim_index
    if current_index >= DomainOutput_l.shape[1]:
        current_index = DomainOutput_l.shape[1] - 1
    displayed_time = current_index * dt_sim
    t_data = np.linspace(0, displayed_time, current_index + 1)
    
    # --- Downsample time series data for realtime plotting ---
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
    
    # Velocity Distribution Graph
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
        yaxis_title="Velocity [m/s]",
        yaxis_range=[v_min, v_max]  # Global bounds for velocity
    )
    
    # Stress Distribution Graph
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(
        x=y_l.flatten(),
        y=DomainOutput_l[:, current_index, 1],
        mode="lines+markers",
        name="Left Domain"
    ))
    fig_stress.add_trace(go.Scatter(
        x=y_r.flatten(),
        y=DomainOutput_r[:, current_index, 1],
        mode="lines+markers",
        name="Right Domain"
    ))
    fig_stress.update_layout(
        title=f"Stress Distribution at t = {displayed_time:.2f} s",
        xaxis_title="x [km]",
        yaxis_title="Stress [MPa]",
        yaxis_range=[s_min, s_max]  # Global bounds for stress
    )
    
    # Time Series Graph (Slip, Slip Rate, and Traction)
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
    # Set global bounds for each subplot
    fig_timeseries.update_yaxes(range=[slip_min, slip_max], row=1, col=1)
    fig_timeseries.update_yaxes(range=[sliprate_min, sliprate_max], row=2, col=1)
    fig_timeseries.update_yaxes(range=[traction_min, traction_max], row=3, col=1)
    
    return fig_velocity, fig_stress, fig_timeseries

# Callback 6: Update animation speed by modifying the interval.
@app.callback(
    Output("interval-component", "interval"),
    Input("anim-speed-slider", "value")
)
def update_animation_speed(speed_value):
    return speed_value * 1000

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)