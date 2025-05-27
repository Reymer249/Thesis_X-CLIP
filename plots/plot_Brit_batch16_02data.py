import plotly.graph_objects as go


x = ["noun", "adj", "verb", "adv", "prep"]
y_control = [0.17, 0.24, 0.28, 0.25, 0.31]
y_neg_2 = [0.48, 0.55, 0.39, 0.39, 0.64]
y_neg_2_llm = [0.37, 0.29, 0.28, 0.29, 0.29]
y_neg_2_2 = [0.36, 0.57, 0.36, 0.39, 0.63]
y_neg_2_llm_2_llm = [0.31, 0.26, 0.27, 0.25, 0.3]

trace_control = go.Bar(x=x, y=y_control, name="Control")
trace2 = go.Bar(x=x, y=y_neg_2, name="Neg=2")
trace2_llm = go.Bar(x=x, y=y_neg_2_llm, name="Neg=2 - LLM")
trace2_2 = go.Bar(x=x, y=y_neg_2_2, name="Neg=2")
trace2_llm_2_llm = go.Bar(x=x, y=y_neg_2_llm_2_llm, name="Neg=2, Pos=2 - LLM")

fig = go.Figure(data=[trace_control, trace2, trace2_2, trace2_llm, trace2_llm_2_llm])
fig.update_layout(barmode="group", title="Brittleness on evaluation set (0.2 data, batch=16)",
                  legend_title="Training Condition")

fig.show()
