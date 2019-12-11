import pandas as pd
import plotly
import plotly.graph_objs as go


def plot_ks_auc(df,name):
    ks_difference = df["ks_trains"]-df["ks_tests"]
    auc_difference = df["auc_trains"]-df["auc_tests"]
    fig = go.Figure()
    x = [i for i in range(len(df))]
    fig.add_trace(go.Scatter(x=x, y=df["ks_trains"],
                        mode='lines+markers',
                        name='ks_trains'))

    fig.add_trace(go.Scatter(x=x, y=df["ks_tests"],
                        mode='lines+markers',
                        name='ks_tests'))
    fig.add_trace(go.Scatter(x=x, y=df["auc_trains"],
                        mode='lines+markers',
                        name='auc_trains'))

    fig.add_trace(go.Scatter(x=x, y=df["auc_tests"],
                        mode='lines+markers',
                        name='auc_tests'))

    fig.add_trace(go.Scatter(x=x, y=auc_difference,
                        mode='lines+markers',
                        name='auc_difference'))


    fig.add_trace(go.Scatter(x=x, y=ks_difference,
                        mode='lines+markers',
                        name='ks_difference'))

    plotly.offline.plot(fig,
                        auto_open=True,
                        filename=(name))
