import plotly.graph_objects as go


x = ["noun", "adj", "verb", "adv", "prep"]
y_full_data = [0.324, 0.368, 0.265, 0.264, 0.160]
y_control = [0.253, 0.333, 0.246, 0.252, 0.144]
y_neg_2 = [0.872, 0.827, 0.926, 0.734, 0.713]
y_neg_4 = [0.885, 0.854, 0, 0.747, 0.737]
y_neg_8 = [0.891, 0.870, 0.943, 0.763, 0.745]

trace1 = go.Bar(x=x, y=y_full_data, name="Full Data")
trace2 = go.Bar(x=x, y=y_control, name="0.2 Data")
trace3 = go.Bar(x=x, y=y_neg_2, name="Neg=2, 0.2 Data")
trace4 = go.Bar(x=x, y=y_neg_4, name="Neg=4, 0.2 Data")
trace5 = go.Bar(x=x, y=y_neg_8, name="Neg=8, 0.2 Data")

fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5])
fig.update_layout(barmode="group", title="PosRank on evaluation set (0.2 data, batch=16)")

fig.show()
