import plotly.graph_objects as go
from constants import *
import numpy as np


x = ["noun", "adj", "verb", "adv", "prep"]

y_neg_20_proper_distribution = [100, 55.7, 97.7, 21.5, 91.4]
y_pos_20_proper_distribution = [100, 45.0, 97.5, 19.4, 0]

y_neg_20 = [98.9, 42.3, 94.4, 7.6, 11.0]
y_pos_20 = [100, 0.6, 20.3, 0.1, 0]

y_neg_100 = [100, 55.4, 97.8, 21.0, 89.9]
y_pos_40 = [99.9, 41.0, 97.1, 16.0, 0]

y_neg_5 = [89.6, 23.0, 82.3, 2.6, 0.3]
y_pos_5 = [100, 0, 0.6, 0, 0]

prop_chen = np.array([1.00, 0.5, 0.99, 0.25, 0.89]) * 100

y_neg = y_neg_5
y_pos = y_pos_5

# Create the figure
fig = go.Figure()

# Add the first bar trace (negative distribution)
fig.add_trace(go.Bar(
    x=x,
    y=y_neg,
    name='Hard Negatives<br>(Set Size 5)',
    marker_color=colors[4]
))

# Add the second bar trace (positive distribution)
fig.add_trace(go.Bar(
    x=x,
    y=y_pos,
    name='Hard Positives<br>(Set Size 5)',
    marker_color=colors[0]
))

fig.add_trace(
    go.Bar(
        x=x,
        y=prop_chen,
        name="Sentence Proportion<br>by Chen et al.",
        marker_color=colors[2]
    )
)

# Update layout
fig.update_layout(
    barmode="group",
    title={
        "text": "Coverage by Part of Speech",
        "font": {"size": title_font},
        "y": title_y,
        "x": 0.02
    },
    legend={
        "font": {"size": legend_font_size},
    },
    xaxis=dict(title='Part of Speech', title_font={"size": xaxis_title_size}, tickfont={"size": xaxis_tickfont_size}),
    yaxis=dict(title='Coverage (%)',title_font={"size": yaxis_title_size}, tickfont={"size": yaxis_tickfont_size}),
    height=1080,
    width=3560
)

# Show the plot
fig.write_image("pos_distribution.pdf")
fig.show()
