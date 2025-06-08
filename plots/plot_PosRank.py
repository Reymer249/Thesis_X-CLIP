import plotly.graph_objects as go
from constants import *


x = ["noun", "adj", "verb", "adv", "prep"]
y_chen_coarse = [0.312, 0.366, 0.276, 0.270, 0.166]
y_chen_hard_neg = [0.875, 0.876, 0.911, 0.811, 0.720]
y_batch_64_full_data = [0.314, 0.347, 0.271, 0.263, 0.154]
y_batch_16_full_data = [0.324, 0.368, 0.265, 0.264, 0.160]
y_batch_16_02data = [0.253, 0.333, 0.246, 0.252, 0.144]
y_neg_2 = [0.872, 0.827, 0.926, 0.734, 0.713]
y_neg_2_llm = [0.681, 0.517, 0.379, 0.331, 0.222]
y_neg_4 = [0.885, 0.854, 0.938, 0.747, 0.737]
y_neg_8 = [0.891, 0.870, 0.943, 0.763, 0.745]

y_neg_2_pos_2 = [0.879, 0.843, 0.928, 0.725, 0.679]
y_neg_2_pos_2_llm = [0.732, 0.511, 0.417, 0.354, 0.229]

y_neg_2_pos_2_set20 = [0.877, 0.761, 0.909, 0.618, 0.570]
y_neg_1_pos_2_set20 = [0.262, 0.366, 0.281, 0.276, 0.166]
y_neg_2_pos_1_set20 = [0.878, 0.740, 0.904, 0.606, 0.560]
y_neg_1_pos_1_set20 = [0.259, 0.330, 0.245, 0.248, 0.159]
y_neg_2_pos_2_set5 = [0.856, 0.737, 0.814, 0.539, 0.454]
y_neg_1_pos_1_set5 = [0.283, 0.331, 0.247, 0.265, 0.167]

chen_coarse = go.Bar(x=x, y=y_chen_coarse, name="Batch 64<br>(Original Study;<br> Coarse)", marker_color=colors[0], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
chen_hard_neg = go.Bar(x=x, y=y_chen_hard_neg, name="Batch 64<br>(Original Study;<br> Fine)", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

batch_64_full_data = go.Bar(x=x, y=y_batch_64_full_data, name="Batch 64<br>(Replicated)", marker_color=colors[0], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
batch_16_full_data = go.Bar(x=x, y=y_batch_16_full_data, name="Batch 16<br>(Replicated)", marker_color=colors[0], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
batch_16_02_data = go.Bar(x=x, y=y_batch_16_02data, name="Control<br>(Neg=0, Pos=0)", marker_color=colors[0], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

neg_2 = go.Bar(x=x, y=y_neg_2, name="Neg=2", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_2_llm = go.Bar(x=x, y=y_neg_2_llm, name="Neg=2 - LLM", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_4 = go.Bar(x=x, y=y_neg_4, name="Neg=4", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_8 = go.Bar(x=x, y=y_neg_8, name="Neg=8", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

neg_2_pos_2 = go.Bar(x=x, y=y_neg_2_pos_2, name="Neg=2 (set size 100)<br>Pos=2 (set size 40)", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_2_pos_2_llm = go.Bar(x=x, y=y_neg_2_pos_2_llm, name="Neg=2, Pos=2 - LLM", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

neg_2_pos_2_set20 = go.Bar(x=x, y=y_neg_2_pos_2_set20, name="Neg=2, Pos=2", marker_color=colors[1], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_1_pos_2_set20 = go.Bar(x=x, y=y_neg_1_pos_2_set20, name="Neg=1, Pos=2", marker_color=colors[2], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_2_pos_1_set20 = go.Bar(x=x, y=y_neg_2_pos_1_set20, name="Neg=2, Pos=1", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_1_pos_1_set20 = go.Bar(x=x, y=y_neg_1_pos_1_set20, name="Neg=1, Pos=1", marker_color=colors[4], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_2_pos_2_set5 = go.Bar(x=x, y=y_neg_2_pos_2_set5, name="Neg=2 (set size 5)<br>Pos=2 (set size 5)", marker_color=colors[3], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)
neg_1_pos_1_set5 = go.Bar(x=x, y=y_neg_1_pos_1_set5, name="Neg=1 (set size 5)<br>Pos=1 (set size 5)", marker_color=colors[4], opacity=opacity, marker_line_color=marker_color, marker_line_width=marker_width)

fig = go.Figure(data=[batch_16_02_data, neg_2_pos_2_set20, neg_1_pos_2_set20, neg_2_pos_1_set20, neg_1_pos_1_set20])
fig.update_layout(
    barmode="group",
    title={
        "text": "PosRank on Evaluation Set (0.2 data, batch 16, set 20)",
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
fig.write_image("posrank.svg")
