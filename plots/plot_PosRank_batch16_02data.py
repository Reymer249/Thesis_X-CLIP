import plotly.graph_objects as go


x = ["noun", "adj", "verb", "adv", "prep"]
y_full_data = [0.324, 0.368, 0.265, 0.264, 0.160]
y_control = [0.253, 0.333, 0.246, 0.252, 0.144]
y_neg_2 = [0.872, 0.827, 0.926, 0.734, 0.713]
y_neg_2_llm = [0.681, 0.517, 0.379, 0.331, 0.222]
y_neg_4 = [0.885, 0.854, 0, 0.747, 0.737]
y_neg_8 = [0.891, 0.870, 0.943, 0.763, 0.745]

y_neg_2_pos_2 = [0.879, 0.843, 0.928, 0.725, 0.679]
y_neg_2_pos_2_llm = [0.732, 0.511, 0.417, 0.354, 0.229]

trace1 = go.Bar(x=x, y=y_full_data, name="Full Data")
trace2 = go.Bar(x=x, y=y_control, name="Control")
trace3 = go.Bar(x=x, y=y_neg_2, name="Neg=2")
trace3_llm = go.Bar(x=x, y=y_neg_2_llm, name="Neg=2 - LLM")
trace4 = go.Bar(x=x, y=y_neg_4, name="Neg=4, 0.2 Data")
trace5 = go.Bar(x=x, y=y_neg_8, name="Neg=8, 0.2 Data")

trace6 = go.Bar(x=x, y=y_neg_2_pos_2, name="Neg=2, Pos=2")
trace6_llm = go.Bar(x=x, y=y_neg_2_pos_2_llm, name="Neg=2, Pos=2 - LLM")

fig = go.Figure(data=[trace2, trace3, trace6, trace3_llm, trace6_llm])
fig.update_layout(barmode="group", title="PosRank on evaluation set (0.2 data, batch=16)",
                  legend_title="Training Condition")

fig.show()
