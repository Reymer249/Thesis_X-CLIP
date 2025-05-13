import plotly.graph_objects as go
from plotly.subplots import make_subplots

# X-Axis categories
x_recall = ["R@1", "R@5", "R@10", "Avg. Recall"]
x_rank = ["Mean Rank"]

# Shared group names and colors
group_labels = ["Control", "Neg=2", "Neg=2, Pos=2"]
colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

# V2T Data
v2t_data = {
    "Full Data": ([53.5, 83.9, 90.9], 5.67),
    "Control": ([73.4, 94.4, 97.6], 2.21),
    "Neg=2": ([65.3, 92.9, 97.8], 2.38),
    "Neg=4, 0.2 Data": ([65.0, 93.7, 97.0], 2.23),
    "Neg=8, 0.2 Data": ([62.1, 91.3, 96.2], 2.72),
    "Neg=2, Pos=2": ([62.1, 93.4, 97.9], 2.51),
}

# T2V Data
t2v_data = {
    "Full Data": ([37.2, 68.8, 80.3], 24.0),
    "Control": ([56.1, 85.0, 91.3], 5.19),
    "Neg=2": ([57.1, 85.7, 92.7], 5.63),
    "Neg=4, 0.2 Data": ([57.3, 86.1, 92.7], 5.03),
    "Neg=8, 0.2 Data": ([56.5, 85.9, 92.7], 5.18),
    "Neg=2, Pos=2": ([57.2, 85.7, 92.7], 4.5),
}


# Helper to compute Avg. Recall
def add_avg_recall(values):
    avg = sum(values) / len(values)
    return values + [avg]


# Create a 2x2 subplot grid with column widths ratio of 3:1
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("V2T Recall", "V2T Mean Rank", "T2V Recall", "T2V Mean Rank"),
    column_widths=[0.75, 0.25],
    horizontal_spacing=0.05
)

# Plot traces
for i, label in enumerate(group_labels):
    color = colors[i]
    group_id = label.lower().replace(" ", "_")

    # V2T
    recalls_v2t, mean_rank_v2t = v2t_data[label]
    recalls_v2t = add_avg_recall(recalls_v2t)

    # V2T Recall (row 1, col 1)
    fig.add_trace(
        go.Bar(
            x=x_recall,
            y=recalls_v2t,
            name=label,
            legendgroup=group_id,
            showlegend=True,
            marker_color=color
        ),
        row=1, col=1
    )

    # V2T Mean Rank (row 1, col 2)
    fig.add_trace(
        go.Bar(
            x=x_rank,
            y=[mean_rank_v2t],
            name=label,
            legendgroup=group_id,
            showlegend=False,
            marker_color=color
        ),
        row=1, col=2
    )

    # T2V
    recalls_t2v, mean_rank_t2v = t2v_data[label]
    recalls_t2v = add_avg_recall(recalls_t2v)

    # T2V Recall (row 2, col 1)
    fig.add_trace(
        go.Bar(
            x=x_recall,
            y=recalls_t2v,
            name=label,
            legendgroup=group_id,
            showlegend=False,
            marker_color=color
        ),
        row=2, col=1
    )

    # T2V Mean Rank (row 2, col 2)
    fig.add_trace(
        go.Bar(
            x=x_rank,
            y=[mean_rank_t2v],
            name=label,
            legendgroup=group_id,
            showlegend=False,
            marker_color=color
        ),
        row=2, col=2
    )

# Update layout and axes
fig.update_layout(
    title="Recalls and Mean Rank on Evaluation Set (0.2 Data, Batch=16)",
    barmode="group",
    height=800,
    legend_title="Training Condition"
)

# Update y-axis titles
fig.update_yaxes(title_text="Recall (%)", row=1, col=1)
fig.update_yaxes(title_text="Mean Rank", row=1, col=2)
fig.update_yaxes(title_text="Recall (%)", row=2, col=1)
fig.update_yaxes(title_text="Mean Rank", row=2, col=2)

# Show the figure
fig.show()