import plotly.graph_objects as go


x = ["noun", "adj", "verb", "adv", "prep"]
y_neg_2 = [0.48, 0.55, 0.39, 0.39, 0.64]
y_neg_4 = [0.36, 0.57, 0.36, 0.39, 0.63]

trace3 = go.Bar(x=x, y=y_neg_2, name="Neg=2")
trace4 = go.Bar(x=x, y=y_neg_4, name="Neg=2, Pos=2")

fig = go.Figure(data=[trace3, trace4])
fig.update_layout(barmode="group", title="Brittleness on evaluation set (0.2 data, batch=16)",
                  legend_title="Training Condition")

fig.show()
