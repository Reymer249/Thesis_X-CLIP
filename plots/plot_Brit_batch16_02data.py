import plotly.graph_objects as go
from constants import *


x = ["noun", "adj", "verb", "adv", "prep"]
y_control = [0.17, 0.24, 0.28, 0.25, 0.31]
y_neg_2 = [0.48, 0.55, 0.39, 0.39, 0.64]
y_neg_2_llm = [0.37, 0.29, 0.28, 0.29, 0.29]
y_neg_2_2 = [0.36, 0.57, 0.36, 0.39, 0.63]
y_neg_2_llm_2_llm = [0.31, 0.26, 0.27, 0.25, 0.3]

y_neg_4 = [0.50, 0.57, 0.38, 0.41, 0.65]
y_neg_8 = [0.51, 0.58, 0.37, 0.39, 0.64]

y_neg_2_pos_2_set20 = [0.45, 0.59, 0.48, 0.46, 0.58]
y_neg_1_pos_2_set20 = [0.14, 0.24, 0.29, 0.28, 0.29]
y_neg_2_pos_1_set20 = [0.54, 0.58, 0.47, 0.43, 0.57]
y_neg_1_pos_1_set20 = [0.16, 0.25, 0.29, 0.26, 0.30]
y_neg_2_pos_2_set5 = [0.45, 0.58, 0.46, 0.39, 0.56]
y_neg_1_pos_1_set5 = [0.16, 0.26, 0.29, 0.26, 0.32]

trace_control = go.Bar(x=x, y=y_control, name="Control<br>(Neg=0, Pos=0)", marker_color=colors[0], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace2 = go.Bar(x=x, y=y_neg_2, name="Neg=2", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace2_llm = go.Bar(x=x, y=y_neg_2_llm, name="Neg=2 - LLM", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace2_2 = go.Bar(x=x, y=y_neg_2_2, name="Neg=2 (set size 100)<br>Pos=2 (set size 40)", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace2_llm_2_llm = go.Bar(x=x, y=y_neg_2_llm_2_llm, name="Neg=2, Pos=2 - LLM", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

trace2_2_20 = go.Bar(x=x, y=y_neg_2_llm_2_llm, name="Neg=2 (set size 20)<br>Pos=2 (set size 20)", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace1_1_20 = go.Bar(x=x, y=y_neg_1_pos_1_set20, name="Neg=1, Pos=1;<br>Set size 20", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace2_2_5 = go.Bar(x=x, y=y_neg_1_pos_1_set5, name="Neg=2 (set size 5)<br>Pos=2 (set size 5)", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace1_1_5 = go.Bar(x=x, y=y_neg_1_pos_1_set5, name="Neg=1, Pos=1;<br>Set size 5", marker_color=colors[4], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)


trace4 = go.Bar(x=x, y=y_neg_4, name="Neg=4", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
trace8 = go.Bar(x=x, y=y_neg_8, name="Neg=8", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

fig = go.Figure(data=[trace_control, trace2_2, trace2_2_20, trace2_2_5])
fig.update_layout(
    barmode="group",
    title={
        "text": "Brittleness on Evaluation Set (0.2 data, batch 16)              ",
        "font": {"size": title_font},
        "y": title_y,
        "x": 0.02
    },
    legend={
        "title": {"text": "Training Condition", "font": {"size": legend_title_font_size}},
        "font": {"size": legend_font_size},
    },
    xaxis={
        "tickfont": {"size": xaxis_tickfont_size}  # X-axis tick labels size
    },
    yaxis={
        "tickfont": {"size": yaxis_tickfont_size}  # Y-axis tick labels size
    },
    height=1080,
    width=1920
)

fig.show()
fig.write_image("brittleness.svg")
