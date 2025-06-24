import plotly.graph_objects as go
import numpy as np

from constants import *

x = ['noun', 'adjective', 'verb', 'adverb', 'preposition']
y_neg_100 = np.array([12999539, 1763652, 9315435, 501783, 0])
y_pos_40 = np.array([5266083, 692179, 5736760, 121599, 0])

y_neg_100 = y_neg_100 / np.sum(y_neg_100)
y_pos_40 = y_pos_40 / np.sum(y_pos_40)

fig = go.Figure(data=[
    go.Bar(name='Hard Negatives', x=x, y=y_neg_100, marker_color=colors[4]),
    go.Bar(name='Hard Positives', x=x, y=y_pos_40, marker_color=colors[0]),
])

# fig.update_layout(
#     barmode='group',
#     title='POS Distribution for Hard Negatives (Set 100) and Hard Positives (Set 40)',
#     xaxis_title='Part of Speech',
#     yaxis_title='Count'
# )

fig.update_layout(
    barmode="group",
    title={
        "text": "POS Distribution for Hard Negatives (Set 100) and Hard Positives (Set 40)",
        "font": {"size": title_font},
        "y": title_y,
        "x": 0.02
    },
    legend={
        "title": {"text": "Training Condition", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size},
    },
    xaxis={
        "tickfont": {"size": xaxis_tickfont_size},  # X-axis tick labels size
        "title": "Part of Speech",
    },
    yaxis={
        "tickfont": {"size": yaxis_tickfont_size},  # Y-axis tick labels size
        "title": "Proportion of Changes in a Set",
    },
    height=1080,
    width=1920
)

fig.update_xaxes(title_font={"size": xaxis_title_size}, tickfont={"size": xaxis_tickfont_size})
fig.update_yaxes(title_font={"size": yaxis_title_size}, tickfont={"size": yaxis_tickfont_size})

fig.show()
