import plotly.graph_objects as go
from plotly.subplots import make_subplots
from constants import *

# X-Axis categories
x_recall = ["R@1", "R@5", "R@10", "Avg. Recall"]
x_rank = ["Mean Rank"]

# Shared group names and colors
group_labels = ["Control<br>(Neg=0, Pos=0)", "Neg=2 (set size 100)<br>Pos=2 (set size 40)", "Neg=2 (set size 20)<br>Pos=2 (set size 20)",
                "Neg=2 (set size 5)<br>Pos=2 (set size 5)"]
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
    "Batch 64<br>(Replicated)": ([66.7, 94.1, 97.5], 2.3),
    "Batch 16<br>(Replicated)": ([63.8, 92.8, 97.1], 2.5),
    "Control<br>(Neg=0, Pos=0)": ([58.6, 89.1, 94.1], 3.4),
    "Neg=2": ([50.0, 86.7, 93.9], 3.7),
    "Neg=2 - LLM": ([53.0, 87.5, 93.8], 4.4),
    "Neg=4": ([49.4, 85.1, 93.4], 4.0),
    "Neg=8": ([45.5, 84.0, 92.1], 4.2),
    "Pos=2": ([58.1, 89.4, 94.5], 3.5),
    "Neg=2 (set size 100)<br>Pos=2 (set size 40)": ([49.0, 86.4, 93.8], 4.4),
    "Neg=2, Pos=2 - LLM": ([52.5, 86.7, 93.9], 4.1),
    "Neg=2 (set size 20)<br>Pos=2 (set size 20)": ([48.8, 85.5, 93.1], 4.5),
    "Neg=1, Pos=2": ([57.3, 89.9, 95.3], 3.6),
    "Neg=2, Pos=1": ([47.7, 85.0, 93.4], 4.1),
    "Neg=1, Pos=1": ([56.8, 88.5, 94.5], 3.9),
    "Neg=2 (set size 5)<br>Pos=2 (set size 5)": ([50.4, 86.5, 93.5], 4.2),
    "Neg=1, Pos=1;<br>Set size 5": ([56.6, 88.4, 94.4], 3.5),

}

# T2V Data
t2v_data = {
    "Batch 64<br>(Original Study;<br>Coarse)": ([0, 0, 0, 82.3], 0),
    "Batch 64<br>(Original Study;<br>Fine)": ([0, 0, 0, 78.2], 0),
    "Batch 64<br>(Replicated)": ([49.6, 83.5, 91.2], 6.9),
    "Batch 16<br>(Replicated)": ([47.3, 82.4, 90.4], 10.9),
    "Control<br>(Neg=0, Pos=0)": ([42.7, 76.3, 86.2], 9.6),
    "Neg=2": ([42.5, 77.0, 86.5], 10.4),
    "Neg=2 - LLM": ([43.0, 76.6, 86.4], 9.5),
    "Neg=4": ([42.7, 77.2, 86.8], 10.2),
    "Neg=8": ([42.7, 77.0, 86.5], 9.8),
    "Pos=2": ([43.4, 77.4, 86.8], 8.9),
    "Neg=2 (set size 100)<br>Pos=2 (set size 40)": ([43.2, 77.1, 86.5], 8.6),
    "Neg=2, Pos=2 - LLM": ([43.9, 77.7, 87.1], 8.7),
    "Neg=2 (set size 20)<br>Pos=2 (set size 20)": ([43.1, 76.5, 86.1], 9.1),
    "Neg=1, Pos=2": ([43.7, 77.7, 87.1], 9.6),
    "Neg=2, Pos=1": ([42.7, 76.6, 86.2], 9.5),
    "Neg=1, Pos=1": ([43.2, 76.9, 86.4], 9.8),
    "Neg=2 (set size 5)<br>Pos=2 (set size 5)": ([42.9, 76.8, 86.5], 9.1),
    "Neg=1, Pos=1;<br>Set size 5": ([43.1, 77.2, 86.4], 10.6),

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
    horizontal_spacing=0.07
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

coef = 0.66

title_font = int(76*coef)
title_y = 0.99
legend_font_size = int(48*coef)
legend_title_font_size = int(56*coef)
xaxis_tickfont_size = int(48*coef)
xaxis_title_size = int(48*coef)
yaxis_tickfont_size = int(48*coef)
yaxis_title_size = int(48*coef)
subplots_name_size = int(48*coef)

fig.update_layout(
    barmode="group",
    title={
        "text": "Recalls and Mean Rank on Evaluation Set (0.2 data, batch=16)",
        "font": {"size": title_font},
        "y": title_y
    },
    legend={
        "title": {"text": "Training Condition    ", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size}
    },
    height=1080,
    width=1920,
    margin=dict(r=300)  # Add this line
)

for annotation in fig['layout']['annotations']:
    annotation['font'] = {'size': subplots_name_size}

for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_font={"size": xaxis_title_size}, tickfont={"size": xaxis_tickfont_size}, row=i, col=j)
        fig.update_yaxes(title_font={"size": yaxis_title_size}, tickfont={"size": yaxis_tickfont_size}, row=i, col=j)

# Show the figure
fig.write_image("recall.svg")