import plotly.graph_objects as go
import torch
import pandas as pd
import os

file = "batches/batch_0_nl_d1cfe577-aad8-4ffd-b238-1c7799fbf5d1.pt"
sample = 0
folder = f"report"

batch = torch.load(file)


nwp_sources = batch["nwp"].keys()

# NWP
for nwp_sources in nwp_sources:
    nwp_data = batch["nwp"][nwp_sources]

    variables = nwp_data["nwp_channel_names"]

    fig = go.Figure()
    init_time = pd.to_datetime(nwp_data["nwp_init_time_utc"][0, 0].numpy(), unit="ns")
    time = init_time + pd.to_timedelta(nwp_data["nwp_step"][0].numpy(), unit="hours")
    for i in range(len(variables)):
        variable = variables[i]
        data = nwp_data["nwp"][0, :, i].mean(dim=[1, 2]).numpy()

        fig.add_trace(go.Scatter(x=time, y=data, mode="lines", name=variable))

    fig.update_layout(
        title=f"{nwp_sources} NWP - example {sample}", xaxis_title="Time", yaxis_title="Value"
    )
    fig.show(renderer='browser')
    name = f"nwp/{nwp_sources}_nwp_{sample}.png"
    fig.write_image(f"{folder}/{name}")

    # heat plot
    fig_heat = go.Figure()
    for i in range(len(variables)):
        fig = go.Figure()
        variable = variables[i]
        data = nwp_data["nwp"][0, 0, i].numpy().T
        fig.add_trace(
            go.Heatmap(
                z=data,
                # x=nwp_data["nwp_latitude"][0].numpy(),
                # y=nwp_data["nwp_latitude"][0].numpy(),
                colorscale="Viridis",
                name=variable,
            )
        )
        fig.update_layout(
            title=f"{nwp_sources} NWP {variable} - example {sample}", xaxis_title="Lon", yaxis_title="Lat"
        )
        name = f"nwp/{nwp_sources}_{variable}_nwp_{sample}.png"
        fig.write_image(f"{folder}/{name}")



# Satellite
satellite_data = batch["satellite_actual"]
fig = go.Figure()
time = pd.to_datetime(batch["satellite_time_utc"][0].numpy(), unit="ns")
for i in range(satellite_data.shape[2]):

    data = satellite_data[0, :, i].mean(dim=[1, 2]).numpy()

    fig.add_trace(go.Scatter(x=time, y=data, mode="lines", name=i))

fig.update_layout(title=f"Satellite - example {sample}", xaxis_title="Time", yaxis_title="Value")
name = f"satellite/satellite_{sample}.png"
fig.write_image(f"{folder}/{name}")

# Site
site_data = batch["site"][0].numpy()
fig = go.Figure()
time = pd.to_datetime(batch["site_time_utc"][0].numpy(), unit="ns")
fig.add_trace(go.Scatter(x=time, y=site_data, mode="lines", name="site"))
fig.update_layout(title=f"Site - example {sample}", xaxis_title="Time", yaxis_title="Value")
name = f"site/site_{sample}.png"
fig.write_image(f"{folder}/{name}")

# time features


fig = go.Figure()

time = pd.to_datetime(batch["site_time_utc"][0].numpy(), unit="ns")
fig.add_trace(go.Scatter(x=time, y=batch["site_time_sin"][0].numpy(), mode="lines", name="time_sin"))
fig.add_trace(go.Scatter(x=time, y=batch["site_time_cos"][0].numpy(), mode="lines", name="time_cos"))
fig.add_trace(go.Scatter(x=time, y=batch["site_date_sin"][0].numpy(), mode="lines", name="date_sin"))
fig.add_trace(go.Scatter(x=time, y=batch["site_date_cos"][0].numpy(), mode="lines", name="date_cos"))
fig.add_trace(go.Scatter(x=time, y=batch["solar_azimuth"][0].numpy(), mode="lines", name="solar_azimuth"))
fig.add_trace(go.Scatter(x=time, y=batch["solar_elevation"][0].numpy(), mode="lines", name="solar_elevation"))
fig.update_layout(title=f"Time features - example {sample}", xaxis_title="Time", yaxis_title="Value")
name = f"site/site_time_{sample}.png"
fig.write_image(f"{folder}/{name}")
