import plotly.graph_objects as go
from constants import *


# V2T Data
v2t_data = {
    "Batch 64<br>(Original Study;<br>Coarse)":([0, 0, 0, 91.1], 0),
    "Batch 64<br>(Original Study;<br>Fine)":([0, 0, 0, 85.3], 0),
    "Batch 64<br>(Replicated)": ([66.7, 94.1, 97.5], 2.3),
    "Batch 16<br>(Replicated)": ([63.8, 92.8, 97.1], 2.5),
    "Batch 16;<br>0.2 data<br>(Replicated)": ([58.6, 89.1, 94.1], 3.4),
    "Batch 16;<br>0.2 data;<br>Neg=2": ([50.0, 86.7, 93.9], 3.7),
    "Neg=2 - LLM": ([53.0, 87.5, 93.8], 4.4),
    "Neg=4": ([49.4, 85.1, 93.4], 4.0),
    "Neg=8": ([45.5, 84.0, 92.1], 4.2),
    "Pos=2": ([58.1, 89.4, 94.5], 3.5),
    "Neg=2, Pos=2": ([49.0, 86.4, 93.8], 4.4),
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
    "Batch 16;<br>0.2 data<br>(Replicated)": ([42.7, 76.3, 86.2], 9.6),
    "Batch 16;<br>0.2 data;<br>Neg=2": ([42.5, 77.0, 86.5], 10.4),
    "Neg=2 - LLM": ([43.0, 76.6, 86.4], 9.5),
    "Neg=4": ([42.7, 77.2, 86.8], 10.2),
    "Neg=8": ([42.7, 77.0, 86.5], 9.8),
    "Pos=2": ([43.4, 77.4, 86.8], 8.9),
    "Neg=2, Pos=2": ([43.2, 77.1, 86.5], 8.6),
    "Neg=2, Pos=2 - LLM": ([43.9, 77.7, 87.1], 8.7),
    "Neg=2 (set size 20)<br>Pos=2 (set size 20)": ([43.1, 76.5, 86.1], 9.1),
    "Neg=1, Pos=2": ([43.7, 77.7, 87.1], 9.6),
    "Neg=2, Pos=1": ([42.7, 76.6, 86.2], 9.5),
    "Neg=1, Pos=1": ([43.2, 76.9, 86.4], 9.8),
    "Neg=2 (set size 5)<br>Pos=2 (set size 5)": ([42.9, 76.8, 86.5], 9.1),
    "Neg=1, Pos=1;<br>Set size 5": ([43.1, 77.2, 86.4], 10.6),
}

# X-axis: V2T and T2V
x = ["V2T Avg. Recall", "T2V Avg. Recall"]

# Shared settings
group_labels = [
    "Batch 64<br>(Original Study;<br>Coarse)",
    "Batch 64<br>(Original Study;<br>Fine)",
    "Batch 16;<br>0.2 data<br>(Replicated)",
    "Batch 16;<br>0.2 data;<br>Neg=2"
]

# Function to compute Avg. Recall
def avg(values):
    return sum(values) / len(values)

# Create figure
fig = go.Figure()

for i, label in enumerate(group_labels):
    color = colors[i]
    group_id = label.lower().replace(" ", "_")

    # Avg Recalls
    if label == "Batch 64<br>(Original Study;<br>Coarse)":
        avg_v2t = 91.1
        avg_t2v = 82.3
    elif label == "Batch 64<br>(Original Study;<br>Fine)":
        avg_v2t = 85.3
        avg_t2v = 78.2
    else:
        avg_v2t = avg(v2t_data[label][0])
        avg_t2v = avg(t2v_data[label][0])

    fig.add_trace(
        go.Bar(
            x=x,
            y=[avg_v2t, avg_t2v],
            name=label,
            legendgroup=group_id,
            marker_color=color,
            opacity=opacity,
            marker_line_color=marker_color,
            marker_line_width=marker_width
        )
    )

# Layout settings
fig.update_layout(
    barmode="group",
    title={
        "text": "Avg. Recall (V2T & T2V)",
        "font": {"size": title_font},
        "y": title_y
    },
    legend={
        "title": {"text": "Training Condition", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size}
    },
    yaxis_title="Avg. Recall (%)",
    xaxis=dict(title_font={"size": xaxis_title_size}, tickfont={"size": xaxis_tickfont_size}),
    yaxis=dict(title_font={"size": yaxis_title_size}, tickfont={"size": yaxis_tickfont_size}),
    height=1080,
    width=1980
)

fig.write_image("recall.svg")
