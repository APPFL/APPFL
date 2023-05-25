import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define event markers
markers = {'expected_arrival': 'bo', 'latest_arrival': 'ro', 'client_arrived': 'mo', 'client_joined': 'co'}

current_dir = os.path.dirname(os.path.abspath(__file__))

log_file = os.path.join(current_dir, "result_server_1.txt")
with open(log_file, 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    match = re.match(r'Group (\d+) (created at (\d+\.\d+) with expected_arrival_time: (\d+\.\d+), latest_arrival_time: (\d+\.\d+)|is deleted at (\d+\.\d+))', line)
    if match is not None:
        group_idx = int(match.group(1))
        if match.group(2).startswith('created'):
            data.append({'time': float(match.group(3)), 'group_idx': group_idx, 'event': 'created'})
            data.append({'time': float(match.group(4)), 'group_idx': group_idx, 'event': 'expected_arrival'})
            data.append({'time': float(match.group(5)), 'group_idx': group_idx, 'event': 'latest_arrival'})
        else:
            data.append({'time': float(match.group(6)), 'group_idx': group_idx, 'event': 'deleted'})
    else:
        match = re.match(r'Client (\d+) (arrived at|joinded) group (\d+) at time (\d+\.\d+)', line)
        if match is not None:
            client_idx = int(match.group(1))
            group_idx = int(match.group(3))
            time = float(match.group(4))
            event = 'client_arrived' if match.group(2) == 'arrived at' else 'client_joined'
            data.append({'time': time, 'group_idx': group_idx, 'event': event, 'client_idx': client_idx})

df = pd.DataFrame(data)

# Create a new figure and a subplot
fig, ax = plt.subplots()

# Plot creation and deletion times
for group_idx in df['group_idx'].unique():
    group_df = df[df['group_idx'] == group_idx]
    creation_time = group_df[group_df['event'] == 'created']['time'].values[0]
    if len(group_df[group_df['event'] == 'deleted']['time'].values) > 0:
        deletion_time = group_df[group_df['event'] == 'deleted']['time'].values[0]
    else:
        deletion_time = group_df[group_df['event'] == 'latest_arrival']['time'].values[0]
    ax.plot([creation_time, deletion_time], [group_idx, group_idx], color='black', linestyle='-', marker='o', markersize=3)

# Determine the maximum active time
max_active_time = df['time'].max()

# Plot the vertical line
ax.axvline(x=max_active_time, color='grey', linestyle='--')

# Plot expected arrival and latest arrival times
expected_arrival_df = df[df['event'] == 'expected_arrival']
ax.scatter(expected_arrival_df['time'], expected_arrival_df['group_idx'], color='blue')

latest_arrival_df = df[df['event'] == 'latest_arrival']
ax.scatter(latest_arrival_df['time'], latest_arrival_df['group_idx'], color='red')

# Plot client joining and arrival times
client_join_df = df[df['event'] == 'client_joined']
ax.scatter(client_join_df['time'], client_join_df['group_idx'], color='green')

client_arrival_df = df[df['event'] == 'client_arrived']
ax.scatter(client_arrival_df['time'], client_arrival_df['group_idx'], color='purple')

from adjustText import adjust_text
texts = []
for _, row in client_arrival_df.iterrows():
    texts.append(ax.text(row['time'], row['group_idx'], f"C{int(row['client_idx'])}", size=8))
for _, row in client_join_df.iterrows():
    texts.append(ax.text(row['time'], row['group_idx'], f"C{int(row['client_idx'])}", size=8))

adjust_text(texts, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            expand_points=(1.2, 1.2),
            expand_text=(1.5, 1.5),
            force_points=(0.1, 0.2),
            force_text=(0, 0.2),
            force_objects=0.1)


# Labeling and formatting
ax.set_xlabel('Time')
ax.set_ylabel('Group index')
ax.yaxis.set_ticks(df['group_idx'].unique())  # show only integer group indices

# Create a legend for the different types of events
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [Line2D([0], [0], color='black', lw=2, label='Group Lifetime'),
                   Line2D([0], [0], color='grey', linestyle='--', lw=2, label='Max Active Time'),
                   Patch(facecolor='blue', label='Expected Arrival Time'),
                   Patch(facecolor='red', label='Latest Arrival Time'),
                   Patch(facecolor='green', label='Client Joined'),
                   Patch(facecolor='purple', label='Client Arrived')]
ax.legend(handles=legend_elements, loc='upper left')

plt.show()   