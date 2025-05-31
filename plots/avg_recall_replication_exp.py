import plotly.graph_objects as go
from constants import *
from plot_recallk_MeanR import v2t_data, t2v_data

# X-axis: V2T and T2V
x = ["V2T Avg. Recall", "T2V Avg. Recall"]

# Shared settings
group_labels = [
    "Batch 64<br>(Original Study;<br>Coarse)",
    "Batch 64<br>(Original Study;<br>Fine)",
    "Batch 16;<br>0.2 data<br>(Replicated)",
    "Batch 16;<br>0.2 data;<br>Neg=2"
]
opacity = 0.75
marker_color = "black"
marker_width = 1
title_font_size = 32
legend_title_font_size = 24
legend_font_size = 20
ticks_font_size = 18
axis_title_size = 20

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
        "font": {"size": title_font_size}
    },
    legend={
        "title": {"text": "Training Condition", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size}
    },
    yaxis_title="Avg. Recall (%)",
    xaxis=dict(title_font={"size": axis_title_size}, tickfont={"size": ticks_font_size}),
    yaxis=dict(title_font={"size": axis_title_size}, tickfont={"size": ticks_font_size}),
    height=1080,
    width=1980
)

fig.show()
