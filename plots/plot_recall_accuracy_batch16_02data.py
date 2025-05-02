import plotly.graph_objects as go
from plotly.subplots import make_subplots

# X-Axis categories
x_recall = ["R@1", "R@5", "R@10", "Avg. Recall"]
x_rank = ["Mean Rank"]

# Shared group names and colors
group_labels = ["Full Data", "0.2 Data", "Neg=2, 0.2 Data", "Neg=4, 0.2 Data", "Neg=8, 0.2 Data"]
colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

# V2T Data
v2t_data = {
    "Full Data": ([53.5, 83.9, 90.9], 5.67),
    "0.2 Data": ([73.4, 94.4, 97.6], 2.21),
    "Neg=2, 0.2 Data": ([65.3, 92.9, 97.8], 2.38),
    "Neg=4, 0.2 Data": ([65.0, 93.7, 97.0], 2.23),
    "Neg=8, 0.2 Data": ([62.1, 91.3, 96.2], 2.72),
}

# T2V Data
t2v_data = {
    "Full Data": ([37.2, 68.8, 80.3], 24.0),
    "0.2 Data": ([56.1, 85.0, 91.3], 5.19),
    "Neg=2, 0.2 Data": ([57.1, 85.7, 92.7], 5.63),
    "Neg=4, 0.2 Data": ([57.3, 86.1, 92.7], 5.03),
    "Neg=8, 0.2 Data": ([56.5, 85.9, 92.7], 5.18),
}

# Helper to compute Avg. Recall
def add_avg_recall(values):
    avg = sum(values) / len(values)
    return values + [avg]

# Create subplot with secondary y-axes
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("V2T Results", "T2V Results"),
    specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
)

# Plot traces
for i, label in enumerate(group_labels):
    color = colors[i]
    group_id = label.lower().replace(" ", "_")

    # V2T
    recalls_v2t, mean_rank_v2t = v2t_data[label]
    recalls_v2t = add_avg_recall(recalls_v2t)

    fig.add_trace(go.Bar(
        x=x_recall,
        y=recalls_v2t,
        name=label,
        legendgroup=group_id,
        showlegend=True,
        marker_color=color
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_rank,
        y=[mean_rank_v2t],
        name=label,
        legendgroup=group_id,
        showlegend=False,
        marker_color=color
    ), row=1, col=1, secondary_y=True)

    # T2V
    recalls_t2v, mean_rank_t2v = t2v_data[label]
    recalls_t2v = add_avg_recall(recalls_t2v)

    fig.add_trace(go.Bar(
        x=x_recall,
        y=recalls_t2v,
        name=label,
        legendgroup=group_id,
        showlegend=False,
        marker_color=color
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_rank,
        y=[mean_rank_t2v],
        name=label,
        legendgroup=group_id,
        showlegend=False,
        marker_color=color
    ), row=2, col=1, secondary_y=True)

# Update layout
fig.update_layout(
    title="Recalls and Mean Rank on Evaluation Set (0.2 Data, Batch=16)",
    barmode="group",
    height=800,
    yaxis_title="Recall (%)",
    yaxis2_title="Mean Rank",
    legend_title="Training Condition"
)

fig.show()
