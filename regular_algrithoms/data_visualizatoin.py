import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import dash
from dash import dcc, html, Input, Output

# 交互式可视化库
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import colorsys
    PLOTLY_AVAILABLE = True
except ImportError:
    print("警告：未安装plotly库。请运行 'pip install plotly' 安装交互式可视化支持。")
    print("将使用matplotlib作为替代方案...")
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

# 1. 数据准备 (与之前相同)
data = {
    "timestamp": "2025-06-06T04:13:20Z",
    "strategy": "balance",
    "sat_attrs": [{"id": 111, "health": 0.9, "pos": [2939.107, 5581.72, 4719.226]},
                  {"id": 112, "health": 0.64, "pos": [-3334.036, 1397.035, 6999.824]},
                  {"id": 113, "health": 0.5, "pos": [-6273.143, -4184.685, 2280.598]},
                  {"id": 114, "health": 0.56, "pos": [-2939.107, -5581.72, -4719.226]},
                  {"id": 115, "health": 1.0, "pos": [3334.036, -1397.035, -6999.824]},
                  {"id": 116, "health": 0.71, "pos": [6273.143, 4184.685, -2280.598]},
                  {"id": 121, "health": 0.73, "pos": [-3552.798, 4281.259, 5577.947]},
                  {"id": 122, "health": 0.91, "pos": [-2447.011, -3445.004, 6648.995]},
                  {"id": 123, "health": 1.0, "pos": [1105.787, -7726.263, 1071.048]},
                  {"id": 124, "health": 0.68, "pos": [3552.798, -4281.259, -5577.947]},
                  {"id": 126, "health": 0.82, "pos": [-1105.787, 7726.263, -1071.048]},
                  {"id": 131, "health": 0.51, "pos": [-4498.06, -1598.404, 6267.184]},
                  {"id": 132, "health": 0.68, "pos": [3009.285, -3980.747, 6096.139]},
                  {"id": 133, "health": 0.72, "pos": [7507.345, -2382.343, -171.046]},
                  {"id": 134, "health": 0.33, "pos": [4498.06, 1598.404, -6267.184]},
                  {"id": 135, "health": 0.59, "pos": [-3009.285, 3980.747, -6096.139]},
                  {"id": 141, "health": 0.94, "pos": [228.012, -4029.186, 6765.997]},
                  {"id": 142, "health": 0.75, "pos": [5546.708, 1609.45, 5358.054]},
                  {"id": 143, "health": 0.63, "pos": [5318.696, 5638.636, -1407.942]},
                  {"id": 151, "health": 0.89, "pos": [3463.991, -482.812, 7059.228]},
                  {"id": 152, "health": 0.69, "pos": [774.356, 6449.734, 4457.168]},
                  {"id": 153, "health": 1.0, "pos": [-2689.635, 6932.546, -2602.059]},
                  {"id": 155, "health": 0.78, "pos": [-774.356, -6449.734, -4457.168]},
                  {"id": 156, "health": 1.0, "pos": [2689.635, -6932.546, 2602.059]},
                  {"id": 161, "health": 0.52, "pos": [859.545, 3221.124, 7137.968]},
                  {"id": 164, "health": 0.69, "pos": [-859.545, -3221.124, -7137.968]},
                  {"id": 166, "health": 0.81, "pos": [6931.324, -452.607, 3717.114]}],
    "sat_edges": [{"from": 111, "to": 112, "w": 0.25}, {"from": 111, "to": 116, "w": 0.25},
                  {"from": 111, "to": 121, "w": 0.29}, {"from": 111, "to": 126, "w": 0.26},
                  {"from": 111, "to": 142, "w": 0.41}, {"from": 111, "to": 143, "w": 0.3},
                  {"from": 111, "to": 151, "w": 0.3}, {"from": 111, "to": 152, "w": 0.83},
                  {"from": 111, "to": 161, "w": 0.49}, {"from": 111, "to": 166, "w": 0.27},
                  {"from": 112, "to": 111, "w": 0.25}, {"from": 112, "to": 113, "w": 0.25},
                  {"from": 112, "to": 121, "w": 0.6}, {"from": 112, "to": 122, "w": 0.39},
                  {"from": 112, "to": 131, "w": 0.59}, {"from": 112, "to": 132, "w": 0.23},
                  {"from": 112, "to": 136, "w": 0.24}, {"from": 112, "to": 141, "w": 0.3},
                  {"from": 112, "to": 142, "w": 0.22}, {"from": 112, "to": 146, "w": 0.21},
                  {"from": 112, "to": 151, "w": 0.28}, {"from": 112, "to": 152, "w": 0.28},
                  {"from": 112, "to": 161, "w": 0.43}, {"from": 112, "to": 162, "w": 0.39},
                  {"from": 113, "to": 112, "w": 0.25}, {"from": 113, "to": 114, "w": 0.25},
                  {"from": 113, "to": 122, "w": 0.33}, {"from": 113, "to": 123, "w": 0.24},
                  {"from": 113, "to": 131, "w": 0.38}, {"from": 113, "to": 136, "w": 0.28},
                  {"from": 113, "to": 141, "w": 0.25}, {"from": 113, "to": 145, "w": 0.24},
                  {"from": 113, "to": 146, "w": 1.0}, {"from": 113, "to": 155, "w": 0.22},
                  {"from": 113, "to": 162, "w": 0.24}, {"from": 113, "to": 163, "w": 0.26},
                  {"from": 114, "to": 113, "w": 0.25}, {"from": 114, "to": 115, "w": 0.25},
                  {"from": 114, "to": 123, "w": 0.26}, {"from": 114, "to": 124, "w": 0.29},
                  {"from": 114, "to": 145, "w": 0.41}, {"from": 114, "to": 146, "w": 0.3},
                  {"from": 114, "to": 154, "w": 0.3}, {"from": 114, "to": 155, "w": 0.83},
                  {"from": 114, "to": 163, "w": 0.27}, {"from": 114, "to": 164, "w": 0.49},
                  {"from": 115, "to": 114, "w": 0.25}, {"from": 115, "to": 116, "w": 0.25},
                  {"from": 115, "to": 124, "w": 0.6}, {"from": 115, "to": 125, "w": 0.39},
                  {"from": 115, "to": 133, "w": 0.24}, {"from": 115, "to": 134, "w": 0.59},
                  {"from": 115, "to": 135, "w": 0.23}, {"from": 115, "to": 143, "w": 0.21},
                  {"from": 115, "to": 144, "w": 0.3}, {"from": 115, "to": 145, "w": 0.22},
                  {"from": 115, "to": 154, "w": 0.28}, {"from": 115, "to": 155, "w": 0.28},
                  {"from": 115, "to": 164, "w": 0.43}, {"from": 115, "to": 165, "w": 0.39},
                  {"from": 116, "to": 111, "w": 0.25}, {"from": 116, "to": 115, "w": 0.25},
                  {"from": 116, "to": 125, "w": 0.33}, {"from": 116, "to": 126, "w": 0.24},
                  {"from": 116, "to": 133, "w": 0.28}, {"from": 116, "to": 134, "w": 0.38},
                  {"from": 116, "to": 142, "w": 0.24}, {"from": 116, "to": 143, "w": 1.0},
                  {"from": 116, "to": 144, "w": 0.25}, {"from": 116, "to": 152, "w": 0.22},
                  {"from": 116, "to": 165, "w": 0.24}, {"from": 116, "to": 166, "w": 0.26},
                  {"from": 121, "to": 111, "w": 0.29}, {"from": 121, "to": 112, "w": 0.6},
                  {"from": 121, "to": 122, "w": 0.25}, {"from": 121, "to": 126, "w": 0.25},
                  {"from": 121, "to": 131, "w": 0.32}, {"from": 121, "to": 136, "w": 0.28},
                  {"from": 121, "to": 141, "w": 0.21}, {"from": 121, "to": 151, "w": 0.23},
                  {"from": 121, "to": 152, "w": 0.39}, {"from": 121, "to": 153, "w": 0.23},
                  {"from": 121, "to": 161, "w": 0.41}, {"from": 121, "to":162, "w": 0.58},
                  {"from": 122, "to": 112, "w": 0.39}, {"from": 122, "to": 113, "w": 0.33},
                  {"from": 122, "to": 121, "w": 0.25}, {"from": 122, "to": 123, "w": 0.25},
                  {"from": 122, "to": 131, "w": 0.7}, {"from": 122, "to": 132, "w": 0.35},
                  {"from": 122, "to": 141, "w": 0.71}, {"from": 122, "to": 146, "w": 0.31},
                  {"from": 122, "to": 151, "w": 0.29}, {"from": 122, "to": 156, "w": 0.26},
                  {"from": 122, "to": 161, "w": 0.26}, {"from": 122, "to": 162, "w": 0.23},
                  {"from": 123, "to": 113, "w": 0.24}, {"from": 123, "to": 114, "w": 0.26},
                  {"from": 123, "to": 122, "w": 0.25}, {"from": 123, "to": 124, "w": 0.25},
                  {"from": 123, "to": 132, "w": 0.3}, {"from": 123, "to": 133, "w": 0.23},
                  {"from": 123, "to": 141, "w": 0.28}, {"from": 123, "to": 146, "w": 0.29},
                  {"from": 123, "to": 155, "w": 0.33}, {"from": 123, "to": 156, "w": 0.83},
                  {"from": 123, "to": 165, "w": 0.25}, {"from": 124, "to": 114, "w": 0.29},
                  {"from": 124, "to": 115, "w": 0.6}, {"from": 124, "to": 123, "w": 0.25},
                  {"from": 124, "to": 125, "w": 0.25}, {"from": 124, "to": 133, "w": 0.28},
                  {"from": 124, "to": 134, "w": 0.32}, {"from": 124, "to": 144, "w": 0.21},
                  {"from": 124, "to": 154, "w": 0.23}, {"from": 124, "to": 155, "w": 0.39},
                  {"from": 124, "to": 156, "w": 0.23}, {"from": 124, "to": 164, "w": 0.41},
                  {"from": 124, "to": 165, "w": 0.58}, {"from": 126, "to": 111, "w": 0.26},
                  {"from": 126, "to": 116, "w": 0.24}, {"from": 126, "to": 121, "w": 0.25},
                  {"from": 126, "to": 125, "w": 0.25}, {"from": 126, "to": 135, "w": 0.3},
                  {"from": 126, "to": 136, "w": 0.23}, {"from": 126, "to": 143, "w": 0.29},
                  {"from": 126, "to": 144, "w": 0.28}, {"from": 126, "to": 152, "w": 0.33},
                  {"from": 126, "to": 153, "w": 0.83}, {"from": 126, "to": 162, "w": 0.25},
                  {"from": 131, "to": 112, "w": 0.59}, {"from": 131, "to": 113, "w": 0.38},
                  {"from": 131, "to": 121, "w": 0.32}, {"from": 131, "to": 122, "w": 0.7},
                  {"from": 131, "to": 132, "w": 0.25}, {"from": 131, "to": 136, "w": 0.25},
                  {"from": 131, "to": 141, "w": 0.36}, {"from": 131, "to": 146, "w": 0.31},
                  {"from": 131, "to": 151, "w": 0.24}, {"from": 131, "to": 161, "w": 0.27},
                  {"from": 131, "to": 162, "w": 0.31}, {"from": 132, "to": 112, "w": 0.23},
                  {"from": 132, "to": 122, "w": 0.35}, {"from": 132, "to": 123, "w": 0.3},
                  {"from": 132, "to": 131, "w": 0.25}, {"from": 132, "to": 133, "w": 0.25},
                  {"from": 132, "to": 141, "w": 0.68}, {"from": 132, "to": 142, "w": 0.31},
                  {"from": 132, "to": 151, "w": 0.53}, {"from": 132, "to": 156, "w": 0.42},
                  {"from": 132, "to": 161, "w": 0.26}, {"from": 132, "to": 166, "w": 0.34},
                  {"from": 133, "to": 115, "w": 0.24}, {"from": 133, "to": 116, "w": 0.28},
                  {"from": 133, "to": 123, "w": 0.23}, {"from": 133, "to": 124, "w": 0.28},
                  {"from": 133, "to": 132, "w": 0.25}, {"from": 133, "to": 134, "w": 0.25},
                  {"from": 133, "to": 142, "w": 0.27}, {"from": 133, "to": 143, "w": 0.23},
                  {"from": 133, "to": 151, "w": 0.23}, {"from": 133, "to": 156, "w": 0.27},
                  {"from": 133, "to": 165, "w": 0.51}, {"from": 133, "to": 166, "w": 0.44},
                  {"from": 134, "to": 115, "w": 0.59}, {"from": 134, "to": 116, "w": 0.38},
                  {"from": 134, "to": 124, "w": 0.32}, {"from": 134, "to": 125, "w": 0.7},
                  {"from": 134, "to": 133, "w": 0.25}, {"from": 134, "to": 135, "w": 0.25},
                  {"from": 134, "to": 143, "w": 0.31}, {"from": 134, "to": 144, "w": 0.36},
                  {"from": 134, "to": 154, "w": 0.24}, {"from": 134, "to": 164, "w": 0.27},
                  {"from": 134, "to": 165, "w": 0.31}, {"from": 135, "to": 115, "w": 0.23},
                  {"from": 135, "to": 125, "w": 0.35}, {"from": 135, "to": 126, "w": 0.3},
                  {"from": 135, "to": 134, "w": 0.25}, {"from": 135, "to": 136, "w": 0.25},
                  {"from": 135, "to": 144, "w": 0.68}, {"from": 135, "to": 145, "w": 0.31},
                  {"from": 135, "to": 153, "w": 0.42}, {"from": 135, "to": 154, "w": 0.53},
                  {"from": 135, "to": 163, "w": 0.34}, {"from": 135, "to": 164, "w": 0.26},
                  {"from": 141, "to": 112, "w": 0.3}, {"from": 141, "to": 113, "w": 0.25},
                  {"from": 141, "to": 121, "w": 0.21}, {"from": 141, "to": 122, "w": 0.71},
                  {"from": 141, "to": 123, "w": 0.28}, {"from": 141, "to": 131, "w": 0.36},
                  {"from": 141, "to": 132, "w": 0.68}, {"from": 141, "to": 142, "w": 0.25},
                  {"from": 141, "to": 146, "w": 0.25}, {"from": 141, "to": 151, "w": 0.4},
                  {"from": 141, "to": 156, "w": 0.34}, {"from": 141, "to": 161, "w": 0.27},
                  {"from": 141, "to": 166, "w": 0.24}, {"from": 142, "to": 111, "w": 0.41},
                  {"from": 142, "to": 112, "w": 0.22}, {"from": 142, "to": 116, "w": 0.24},
                  {"from": 142, "to": 132, "w": 0.31}, {"from": 142, "to": 133, "w": 0.27},
                  {"from": 142, "to": 141, "w": 0.25}, {"from": 142, "to": 143, "w": 0.25},
                  {"from": 142, "to": 151, "w": 0.57}, {"from": 142, "to": 152, "w": 0.28},
                  {"from": 142, "to": 161, "w": 0.37}, {"from": 142, "to": 166, "w": 0.65},
                  {"from": 143, "to": 111, "w": 0.3}, {"from": 143, "to": 115, "w": 0.21},
                  {"from": 143, "to": 116, "w": 1.0}, {"from": 143, "to": 125, "w": 0.31},
                  {"from": 143, "to": 126, "w": 0.29}, {"from": 143, "to": 133, "w": 0.23},
                  {"from": 143, "to": 134, "w": 0.31}, {"from": 143, "to": 142, "w": 0.25},
                  {"from": 143, "to": 144, "w": 0.25}, {"from": 143, "to": 152, "w": 0.26},
                  {"from": 143, "to": 153, "w": 0.24}, {"from": 143, "to": 166, "w": 0.24},
                  {"from": 151, "to": 111, "w": 0.3}, {"from": 151, "to": 112, "w": 0.28},
                  {"from": 151, "to": 121, "w": 0.23}, {"from": 151, "to": 122, "w": 0.29},
                  {"from": 151, "to": 131, "w": 0.24}, {"from": 151, "to": 132, "w": 0.53},
                  {"from": 151, "to": 133, "w": 0.23}, {"from": 151, "to": 141, "w": 0.4},
                  {"from": 151, "to": 142, "w": 0.57}, {"from": 151, "to": 152, "w": 0.25},
                  {"from": 151, "to": 156, "w": 0.25}, {"from": 151, "to": 161, "w": 0.43},
                  {"from": 151, "to": 166, "w": 0.4}, {"from": 152, "to": 111, "w": 0.83},
                  {"from": 152, "to": 112, "w": 0.28}, {"from": 152, "to": 116, "w": 0.22},
                  {"from": 152, "to": 121, "w": 0.39}, {"from": 152, "to": 126, "w": 0.33},
                  {"from": 152, "to": 142, "w": 0.28}, {"from": 152, "to": 143, "w": 0.26},
                  {"from": 152, "to": 151, "w": 0.25}, {"from": 152, "to": 153, "w": 0.25},
                  {"from": 152, "to": 161, "w": 0.46}, {"from": 152, "to": 162, "w": 0.26},
                  {"from": 153, "to": 121, "w": 0.23}, {"from": 153, "to": 125, "w": 0.26},
                  {"from": 153, "to": 126, "w": 0.83}, {"from": 153, "to": 135, "w": 0.42},
                  {"from": 153, "to": 136, "w": 0.27}, {"from": 153, "to": 143, "w": 0.24},
                  {"from": 153, "to": 144, "w": 0.34}, {"from": 153, "to": 152, "w": 0.25},
                  {"from": 153, "to": 154, "w": 0.25}, {"from": 153, "to": 162, "w": 0.25},
                  {"from": 153, "to": 163, "w": 0.25}, {"from": 155, "to": 113, "w": 0.22},
                  {"from": 155, "to": 114, "w": 0.83}, {"from": 155, "to": 115, "w": 0.28},
                  {"from": 155, "to": 123, "w": 0.33}, {"from": 155, "to": 124, "w": 0.39},
                  {"from": 155, "to": 145, "w": 0.28}, {"from": 155, "to": 146, "w": 0.26},
                  {"from": 155, "to": 154, "w": 0.25}, {"from": 155, "to": 156, "w": 0.25},
                  {"from": 155, "to": 164, "w": 0.46}, {"from": 155, "to": 165, "w": 0.26},
                  {"from": 156, "to": 122, "w": 0.26}, {"from": 156, "to": 123, "w": 0.83},
                  {"from": 156, "to": 124, "w": 0.23}, {"from": 156, "to": 132, "w": 0.42},
                  {"from": 156, "to": 133, "w": 0.27}, {"from": 156, "to": 141, "w": 0.34},
                  {"from": 156, "to": 146, "w": 0.24}, {"from": 156, "to": 151, "w": 0.25},
                  {"from": 156, "to": 155, "w": 0.25}, {"from": 156, "to": 165, "w": 0.25},
                  {"from": 156, "to": 166, "w": 0.25}, {"from": 161, "to": 111, "w": 0.49},
                  {"from": 161, "to": 112, "w": 0.43}, {"from": 161, "to": 121, "w": 0.41},
                  {"from": 161, "to": 122, "w": 0.26}, {"from": 161, "to": 131, "w": 0.27},
                  {"from": 161, "to": 132, "w": 0.26}, {"from": 161, "to": 141, "w": 0.27},
                  {"from": 161, "to": 142, "w": 0.37}, {"from": 161, "to": 151, "w": 0.43},
                  {"from": 161, "to": 152, "w": 0.46}, {"from": 161, "to": 162, "w": 0.25},
                  {"from": 161, "to": 166, "w": 0.25}, {"from": 164, "to": 114, "w": 0.49},
                  {"from": 164, "to": 115, "w": 0.43}, {"from": 164, "to": 124, "w": 0.41},
                  {"from": 164, "to": 125, "w": 0.26}, {"from": 164, "to": 134, "w": 0.27},
                  {"from": 164, "to": 135, "w": 0.26}, {"from": 164, "to": 144, "w": 0.27},
                  {"from": 164, "to": 145, "w": 0.37}, {"from": 164, "to": 154, "w": 0.43},
                  {"from": 164, "to": 155, "w": 0.46}, {"from": 164, "to": 163, "w": 0.25},
                  {"from": 164, "to": 165, "w": 0.25}, {"from": 166, "to": 111, "w": 0.27},
                  {"from": 166, "to": 116, "w": 0.26}, {"from": 166, "to": 132, "w": 0.34},
                  {"from": 166, "to": 133, "w": 0.44}, {"from": 166, "to": 141, "w": 0.24},
                  {"from": 166, "to": 142, "w": 0.65}, {"from": 166, "to": 143, "w": 0.24},
                  {"from": 166, "to": 151, "w": 0.4}, {"from": 166, "to": 156, "w": 0.25},
                  {"from": 166, "to": 161, "w": 0.25}, {"from": 166, "to": 165, "w": 0.25}],
    "target_edges": [{"from": 111, "to": 11, "q": 0.17}, {"from": 111, "to": 24, "q": 0.15},
                     {"from": 111, "to": 25, "q": 0.19}, {"from": 111, "to": 30, "q": 0.15},
                     {"from": 111, "to": 40, "q": 0.17}, {"from": 111, "to": 44, "q": 0.16},
                     {"from": 111, "to": 45, "q": 0.15}, {"from": 111, "to": 47, "q": 0.15},
                     {"from": 111, "to": 48, "q": 1.0}, {"from": 112, "to": 18, "q": 0.2},
                     {"from": 112, "to": 23, "q": 0.18}, {"from": 112, "to": 33, "q": 0.17},
                     {"from": 112, "to": 46, "q": 0.21}, {"from": 113, "to": 19, "q": 0.13},
                     {"from": 113, "to": 37, "q": 0.17}, {"from": 114, "to": 19, "q": 0.14},
                     {"from": 115, "to": 29, "q": 0.19}, {"from": 116, "to": 11, "q": 0.13},
                     {"from": 116, "to": 14, "q": 0.13}, {"from": 121, "to": 24, "q": 0.16},
                     {"from": 121, "to": 25, "q": 0.14}, {"from": 121, "to": 40, "q": 0.15},
                     {"from": 121, "to": 46, "q": 0.28}, {"from": 122, "to": 13, "q": 0.16},
                     {"from": 122, "to": 15, "q": 0.13}, {"from": 122, "to": 16, "q": 0.63},
                     {"from": 122, "to": 18, "q": 0.17}, {"from": 122, "to": 23, "q": 0.25},
                     {"from": 122, "to": 35, "q": 0.17}, {"from": 122, "to": 50, "q": 0.17},
                     {"from": 123, "to": 12, "q": 0.13}, {"from": 123, "to": 17, "q": 0.31},
                     {"from": 123, "to": 26, "q": 0.13}, {"from": 123, "to": 6, "q": 0.14},
                     {"from": 124, "to": 29, "q": 0.18}, {"from": 124, "to": 7, "q": 0.27},
                     {"from": 126, "to": 38, "q": 0.18}, {"from": 131, "to": 16, "q": 0.19},
                     {"from": 131, "to": 23, "q": 0.19}, {"from": 132, "to": 13, "q": 0.17},
                     {"from": 132, "to": 15, "q": 0.14}, {"from": 132, "to": 26, "q": 0.23},
                     {"from": 132, "to": 3, "q": 0.42}, {"from": 132, "to": 36, "q": 0.15},
                     {"from": 132, "to": 43, "q": 0.46}, {"from": 132, "to": 50, "q": 0.23},
                     {"from": 132, "to": 6, "q": 0.2}, {"from": 132, "to": 8, "q": 0.19},
                     {"from": 133, "to": 10, "q": 0.18}, {"from": 133, "to": 14, "q": 0.18},
                     {"from": 134, "to": 29, "q": 0.15}, {"from": 135, "to": 21, "q": 0.13},
                     {"from": 141, "to": 13, "q": 0.24}, {"from": 141, "to": 15, "q": 0.16},
                     {"from": 141, "to": 2, "q": 0.17}, {"from": 141, "to": 23, "q": 0.18},
                     {"from": 141, "to": 26, "q": 0.21}, {"from": 141, "to": 27, "q": 0.13},
                     {"from": 141, "to": 3, "q": 0.24}, {"from": 141, "to": 34, "q": 0.2},
                     {"from": 141, "to": 36, "q": 0.17}, {"from": 141, "to": 43, "q": 0.16},
                     {"from": 141, "to": 49, "q": 0.16}, {"from": 141, "to": 50, "q": 0.48},
                     {"from": 142, "to": 1, "q": 0.17}, {"from": 142, "to": 10, "q": 0.13},
                     {"from": 142, "to": 20, "q": 0.23}, {"from": 142, "to": 27, "q": 0.15},
                     {"from": 142, "to": 30, "q": 0.21}, {"from": 142, "to": 39, "q": 0.15},
                     {"from": 142, "to": 47, "q": 0.16}, {"from": 143, "to": 11, "q": 0.17},
                     {"from": 143, "to": 44, "q": 0.19}, {"from": 151, "to": 13, "q": 0.19},
                     {"from": 151, "to": 15, "q": 0.23}, {"from": 151, "to": 2, "q": 0.18},
                     {"from": 151, "to": 20, "q": 0.17}, {"from": 151, "to": 3, "q": 0.2},
                     {"from": 151, "to": 30, "q": 0.21}, {"from": 151, "to": 34, "q": 0.26},
                     {"from": 151, "to": 36, "q": 0.26}, {"from": 151, "to": 45, "q": 0.14},
                     {"from": 151, "to": 47, "q": 0.19}, {"from": 151, "to": 49, "q": 0.2},
                     {"from": 152, "to": 24, "q": 0.34}, {"from": 152, "to": 25, "q": 0.59},
                     {"from": 152, "to": 33, "q": 0.15}, {"from": 152, "to": 40, "q": 0.2},
                     {"from": 152, "to": 45, "q": 0.14}, {"from": 152, "to": 46, "q": 0.13},
                     {"from": 152, "to": 48, "q": 0.22}, {"from": 153, "to": 38, "q": 0.15},
                     {"from": 155, "to": 7, "q": 0.2}, {"from": 156, "to": 12, "q": 0.19},
                     {"from": 156, "to": 17, "q": 0.15}, {"from": 156, "to": 26, "q": 0.2},
                     {"from": 156, "to": 43, "q": 0.17}, {"from": 156, "to": 50, "q": 0.13},
                     {"from": 156, "to": 6, "q": 0.28}, {"from": 156, "to": 8, "q": 0.14},
                     {"from": 161, "to": 15, "q": 0.14}, {"from": 161, "to": 24, "q": 0.13},
                     {"from": 161, "to": 25, "q": 0.15}, {"from": 161, "to": 28, "q": 0.16},
                     {"from": 161, "to": 30, "q": 0.21}, {"from": 161, "to": 36, "q": 0.14},
                     {"from": 161, "to": 40, "q": 0.34}, {"from": 161, "to": 46, "q": 0.21},
                     {"from": 161, "to": 47, "q": 0.31}, {"from": 161, "to": 48, "q": 0.13},
                     {"from": 164, "to": 7, "q": 0.14}, {"from": 166, "to": 1, "q": 0.18},
                     {"from": 166, "to": 10, "q": 0.25}, {"from": 166, "to": 14, "q": 0.15},
                     {"from": 166, "to": 27, "q": 0.15}, {"from": 166, "to": 39, "q": 0.16},
                     {"from": 166, "to": 8, "q": 0.14}]
}

# 2. 构建图 (与之前相同)
G = nx.Graph()

# 添加卫星节点
sat_nodes_list = []
for sat in data['sat_attrs']:
    node_id = f"s{sat['id']}"
    G.add_node(node_id, type='satellite', health=sat['health'], pos=np.array(sat['pos']))
    sat_nodes_list.append(node_id)

# 添加目标节点
for edge in data['target_edges']:
    G.add_node(f"t{edge['to']}", type='target')

# 添加卫星-卫星边
for edge in data['sat_edges']:
    if f"s{edge['from']}" in G and f"s{edge['to']}" in G:
        G.add_edge(f"s{edge['from']}", f"s{edge['to']}", weight=1.0 / (edge['w'] + 1e-6), type='sat_link')

# 添加卫星-目标边
for edge in data['target_edges']:
    if f"s{edge['from']}" in G and f"t{edge['to']}" in G:
        G.add_edge(f"s{edge['from']}", f"t{edge['to']}", weight=1.0 / (edge['q'] + 1e-6), type='target_link')

# 删除簇相关内容，直接进行交互式可视化

# 5. 交互式可视化准备
# === 基于物理位置的布局 ===
# 提取所有卫星的3D坐标
sat_3d_pos = np.array([G.nodes[node]['pos'] for node in sat_nodes_list])

# 使用PCA将3D坐标降至2D
pca = PCA(n_components=2)
sat_2d_pos_transformed = pca.fit_transform(sat_3d_pos)

# 创建一个只包含卫星2D位置的初始布局字典
pos_init = {node: sat_2d_pos_transformed[i] for i, node in enumerate(sat_nodes_list)}

# 首先只布局卫星节点，大幅增大k值让卫星最大程度分散
sat_subgraph = G.subgraph(sat_nodes_list)
pos_sat_only = nx.spring_layout(sat_subgraph, pos=pos_init, k=400, iterations=2000, seed=42)

# 为所有节点创建位置字典，先放入卫星位置
pos = {}
for sat_node in sat_nodes_list:
    pos[sat_node] = pos_sat_only[sat_node]

# 节点类型和属性
node_types = nx.get_node_attributes(G, 'type')

# 计算卫星网络的边界范围
sat_positions = np.array([pos[sat] for sat in sat_nodes_list])
sat_center = np.mean(sat_positions, axis=0)
sat_radius = np.max([np.linalg.norm(pos[sat] - sat_center) for sat in sat_nodes_list])

# 为每个目标节点手动分配位置，减少距离让它们更接近卫星
target_nodes = [node for node in G.nodes() if node_types[node] == 'target']
outer_radius = sat_radius + 1.2  # 在卫星网络外围1.2个单位

# 将所有目标节点均匀分布在外围圆周上
for i, target_node in enumerate(target_nodes):
    angle = (i * 2 * np.pi) / len(target_nodes)
    direction = np.array([np.cos(angle), np.sin(angle)])
    pos[target_node] = sat_center + direction * outer_radius

# 为每个节点分配唯一颜色（用于其出边）- 降低饱和度的版本
all_nodes = list(G.nodes())
def reduce_saturation(hex_color, saturation_factor=0.6):
    """降低颜色饱和度"""
    # 处理不同的颜色格式
    if hex_color.startswith('rgb'):
        import re
        rgb_values = re.findall(r'\d+', hex_color)
        if len(rgb_values) >= 3:
            rgb = (int(rgb_values[0])/255.0, int(rgb_values[1])/255.0, int(rgb_values[2])/255.0)
        else:
            rgb = (0.5, 0.5, 0.5)
    elif hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        else:
            rgb = (0.5, 0.5, 0.5)
    else:
        try:
            import matplotlib.colors as mcolors
            hex_color = mcolors.to_hex(hex_color)
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        except:
            rgb = (0.5, 0.5, 0.5)
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    s = s * saturation_factor
    rgb_new = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_new[0]*255), int(rgb_new[1]*255), int(rgb_new[2]*255))

if PLOTLY_AVAILABLE:
    px_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    node_edge_color_map = {}
    for i, node in enumerate(all_nodes):
        original_color = px_colors[i % len(px_colors)]
        node_edge_color_map[node] = reduce_saturation(original_color, 0.5)
else:
    import matplotlib.colors as mcolors
    node_colors_for_edges = plt.cm.get_cmap('tab20', len(all_nodes))
    node_edge_color_map = {}
    for i, node in enumerate(all_nodes):
        original_color = mcolors.to_hex(node_colors_for_edges(i))
        node_edge_color_map[node] = reduce_saturation(original_color, 0.5)

# 6. 创建可视化图表
# 准备边数据
sat_edges = [edge for edge in G.edges(data=True) if edge[2]['type'] == 'sat_link']
target_edges = [edge for edge in G.edges(data=True) if edge[2]['type'] == 'target_link']

# Dash 应用
app = dash.Dash(__name__)

def make_figure(highlight_node=None):
    """生成图表，若指定节点则高亮相关连线
    Args:
        highlight_node (str): 被高亮的节点id（如's111'或't11'），为None时无高亮
    Returns:
        go.Figure: plotly图对象
    """
    fig = go.Figure()
    # --- 画卫星间边 ---
    for edge in sat_edges:
        source, target, attr = edge
        color = '#e74c3c' if highlight_node and (source == highlight_node or target == highlight_node) else '#cccccc'
        width = 4 if highlight_node and (source == highlight_node or target == highlight_node) else 2
        fig.add_trace(go.Scatter(
            x=[pos[source][0], pos[target][0]],
            y=[pos[source][1], pos[target][1]],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none',
            showlegend=False,
            name='卫星间连接'
        ))
    # --- 画卫星-目标边 ---
    for edge in target_edges:
        source, target, attr = edge
        color = '#2980b9' if highlight_node and (source == highlight_node or target == highlight_node) else '#bbbbbb'
        width = 3 if highlight_node and (source == highlight_node or target == highlight_node) else 1.5
        fig.add_trace(go.Scatter(
            x=[pos[source][0], pos[target][0]],
            y=[pos[source][1], pos[target][1]],
            mode='lines',
            line=dict(width=width, color=color, dash='dash'),
            hoverinfo='none',
            showlegend=False,
            name='卫星-目标连接'
        ))
    # --- 画卫星节点 ---
    for node in sat_nodes_list:
        x, y = pos[node]
        color = '#f39c12' if highlight_node == node else 'green'
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(width=3, color='white')),
            text=[node.replace('s1', '')],
            textposition="middle center",
            hoverinfo='text',
            name='卫星节点',
            customdata=[node],
            showlegend=False
        ))
    # --- 画目标节点 ---
    for node in target_nodes:
        x, y = pos[node]
        color = '#8e44ad' if highlight_node == node else '#D3D3D3'
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=16, color=color, symbol='square', line=dict(width=2, color='#696969')),
            text=[node.replace('t', 'T')],
            textposition="middle center",
            hoverinfo='text',
            name='目标节点',
            customdata=[node],
            showlegend=False
        ))
    fig.update_layout(
        title='卫星网络连接关系与目标可见性（点击节点高亮）',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(b=80, l=60, r=60, t=100),
        dragmode='pan'
    )
    return fig

app.layout = html.Div([
    html.H2("卫星网络交互式图表（点击节点高亮相关连线）"),
    dcc.Graph(
        id='network-graph',
        figure=make_figure(),
        config={'displayModeBar': True, 'scrollZoom': True},
        style={'height': '95vh', 'width': '100%'}
    ),
    html.Div("点击任意卫星或目标节点，高亮其相关连线。再次点击空白处恢复。", style={'color': 'gray'})
])

@app.callback(
    Output('network-graph', 'figure'),
    Input('network-graph', 'clickData')
)
def update_highlight(clickData):
    """Dash回调：根据点击事件高亮相关连线
    Args:
        clickData (dict): dash点击事件数据
    Returns:
        go.Figure: 更新后的图
    """
    if clickData and clickData['points']:
        node = clickData['points'][0]['customdata']
        return make_figure(highlight_node=node)
    return make_figure()

if __name__ == '__main__':
    app.run(debug=True)

# 显示网络信息
print("卫星网络可视化完成")
print(f"卫星节点数: {len([n for n in G.nodes() if node_types[n] == 'satellite'])}")
print(f"目标节点数: {len([n for n in G.nodes() if node_types[n] == 'target'])}")
print(f"卫星间连接数: {len(sat_edges)}")
print(f"卫星-目标连接数: {len(target_edges)}")

