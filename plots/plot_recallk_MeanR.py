import plotly.graph_objects as go
from plotly.subplots import make_subplots
from constants import *

# X-Axis categories
x_recall = ["R@1", "R@5", "R@10", "Avg. Recall"]
x_rank = ["Mean Rank"]

# Shared group names and colors
group_labels = ["Batch 64<br>(Original Study;<br>Coarse)", "Batch 64<br>(Replicated)", "Batch 16<br>(Replicated)"]
opacity = 0.75
marker_color = "black"
marker_width = 1
title_font_size = 36
title_font_size_x = 20
title_font_size_y = 20
ticks_font_size_x = 24
ticks_font_size_y = 24
legend_title_font_size = 28
legend_font_size = 24
subplots_name_size = 24

# V2T Data
v2t_data = {
    "Batch 64<br>(Original Study;<br>Coarse)":([0, 0, 0, 91.1], 0),
    "Batch 64<br>(Original Study;<br>Fine)":([0, 0, 0, 85.3], 0),
    "Batch 64<br>(Replicated)": ([55.2, 85.1, 91.8], 5.05),
    "Batch 16<br>(Replicated)": ([53.5, 83.9, 90.9], 5.67),
    "Batch 16;<br>0.2 data<br>(Replicated)": ([73.4, 94.4, 97.6], 2.21),
    "Batch 16;<br>0.2 data;<br>Neg=2": ([65.3, 92.9, 97.8], 2.38),
    "Neg=4, 0.2 Data": ([65.0, 93.7, 97.0], 2.23),
    "Neg=8, 0.2 Data": ([62.1, 91.3, 96.2], 2.72),
    "Neg=2, Pos=2": ([62.1, 93.4, 97.9], 2.51),
}

# T2V Data
t2v_data = {
    "Batch 64<br>(Original Study;<br>Coarse)": ([0, 0, 0, 82.3], 0),
    "Batch 64<br>(Original Study;<br>Fine)": ([0, 0, 0, 78.2], 0),
    "Batch 64<br>(Replicated)": ([39.0, 70.2, 81.5], 15.33),
    "Batch 16<br>(Replicated)": ([37.2, 68.8, 80.3], 24.0),
    "Batch 16;<br>0.2 data<br>(Replicated)": ([56.1, 85.0, 91.3], 5.19),
    "Batch 16;<br>0.2 data;<br>Neg=2": ([57.1, 85.7, 92.7], 5.63),
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
            marker_color=color,
            opacity=opacity,
            marker_line_color=marker_color,
            marker_line_width=marker_width
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
            marker_color=color,
            opacity=opacity,
            marker_line_color=marker_color,
            marker_line_width=marker_width
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
            marker_color=color,
            opacity=opacity,
            marker_line_color=marker_color,
            marker_line_width=marker_width
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
            marker_color=color,
            opacity=opacity,
            marker_line_color=marker_color,
            marker_line_width=marker_width
        ),
        row=2, col=2
    )

# Update y-axis titles
fig.update_yaxes(title_text="Recall (%)", row=1, col=1)
fig.update_yaxes(title_text="Mean Rank", row=1, col=2)
fig.update_yaxes(title_text="Recall (%)", row=2, col=1)
fig.update_yaxes(title_text="Mean Rank", row=2, col=2)

fig.update_layout(
    barmode="group",
    title={
        "text": "Recalls and Mean Rank on Evaluation Set (0.2 Data, Batch=16)",
        "font": {"size": title_font_size}
    },
    legend={
        "title": {"text": "Training Condition", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size}
    },
    height=1080,
    width=1920
)

for annotation in fig['layout']['annotations']:
    annotation['font'] = {'size': subplots_name_size}

for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_font={"size": title_font_size_x}, tickfont={"size": ticks_font_size_x}, row=i, col=j)
        fig.update_yaxes(title_font={"size": title_font_size_y}, tickfont={"size": ticks_font_size_y}, row=i, col=j)

# Show the figure
fig.show()