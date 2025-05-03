import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np


path=r"app\data\cleaned\Wind farm site 1 (Nominal capacity-99MW)1746191929.0945573.csv"
# 读取数据并解析时间
df = pd.read_csv(path)
# df = df.sort_values('Time(year-month-day h:m:s)')  # 确保按时间排序
df = df.iloc[:100,:]

# 提取时间和功率数据
time = df.iloc[:, 0]  # 假设第一列是时间
power = df.iloc[:, -1]  # 假设最后一列是功率


def plot_data(path):
    df = pd.read_csv(path)
    df=df.iloc[:100,:]
    time = df.iloc[:, 0]
    power = df.iloc[:, -1]
# LOWESS拟合（用数据索引作为等间距x值）
    x = np.arange(len(time))
    lowess = sm.nonparametric.lowess(power, x, frac=0.05)
    fitted_power = lowess[:, 1]

    # 创建动态折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=power,
        mode='lines',
        name='实际功率',
        line=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=time,
        y=fitted_power,
        mode='lines',
        name='LOWESS拟合',
        line=dict(color='#ff7f0e', dash='dot')
    ))

    # 设置图表布局
    fig.update_layout(
        title='风电场发电功率趋势分析',
        xaxis_title='时间',
        yaxis_title='功率 (MW)',
        hovermode='x unified',
        template='plotly_white'
    )

# # 保存为独立HTML文件
    fig.write_html('power_analysis.html')   

plot_data(path)