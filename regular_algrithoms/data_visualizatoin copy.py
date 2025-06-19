import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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

if PLOTLY_AVAILABLE:
    # 使用plotly创建交互式图表
    fig = go.Figure()
    
    # 按源节点分组绘制卫星间边，每个卫星用不同颜色
    sat_edges_by_source = {}
    for edge in sat_edges:
        source = edge[0]
        if source not in sat_edges_by_source:
            sat_edges_by_source[source] = []
        sat_edges_by_source[source].append(edge)
    for source_node, edges in sat_edges_by_source.items():
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_label = source_node.replace('s1', '')
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                 line=dict(width=2, color=node_edge_color_map[source_node]),
                                 hoverinfo='none',
                                 mode='lines',
                                 name=f'卫星{node_label}间连接',
                                 showlegend=False))
    
    # 按源节点分组绘制卫星-目标边，每个卫星用不同颜色
    target_edges_by_source = {}
    for edge in target_edges:
        source = edge[0]
        if source not in target_edges_by_source:
            target_edges_by_source[source] = []
        target_edges_by_source[source].append(edge)
    for source_node, edges in target_edges_by_source.items():
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_label = source_node.replace('s1', '')
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                 line=dict(width=1.5, color=node_edge_color_map[source_node], dash='dash'),
                                 hoverinfo='none',
                                 mode='lines',
                                 name=f'卫星{node_label}目标连接',
                                 showlegend=False))
    
    # 添加卫星节点
    sat_node_x = []
    sat_node_y = []
    sat_node_text = []
    sat_node_sizes = []
    sat_node_colors = []
    sat_node_hover = []
    
    for node in sat_nodes_list:
        x, y = pos[node]
        sat_node_x.append(x)
        sat_node_y.append(y)
        # 简化节点标签
        node_label = node.replace('s1', '')
        sat_node_text.append(node_label)
        pos_str = f"[{G.nodes[node]['pos'][0]:.1f}, {G.nodes[node]['pos'][1]:.1f}, {G.nodes[node]['pos'][2]:.1f}]"
        node_text = f'卫星 {node_label}<br>健康度: {G.nodes[node]["health"]:.2f}<br>位置: {pos_str}'
        sat_node_hover.append(node_text)
        # 根据健康度设置大小和颜色
        health = G.nodes[node]['health']
        sat_node_sizes.append(health * 30 + 15)  # 大小范围15-45
        sat_node_colors.append(health)  # 颜色基于健康度
    
    fig.add_trace(go.Scatter(x=sat_node_x, y=sat_node_y,
                              mode='markers+text',
                              marker=dict(size=sat_node_sizes,
                                        color=sat_node_colors,
                                        colorscale='Viridis',
                                        line=dict(width=3, color='white'),
                                        showscale=True,
                                        colorbar=dict(title="健康度")),
                              text=sat_node_text,
                              textposition="middle center",
                              textfont=dict(size=10, color='white', family='Arial Black'),
                              hovertext=sat_node_hover,
                              hoverinfo='text',
                              name='卫星节点',
                              showlegend=True))
    
    # 添加目标节点
    target_node_x = []
    target_node_y = []
    target_node_text = []
    target_node_ids = []
    
    for node in target_nodes:
        x, y = pos[node]
        target_node_x.append(x)
        target_node_y.append(y)
        target_node_text.append(f'目标 {node.replace("t", "T")}')
        target_node_ids.append(node)
    
    fig.add_trace(go.Scatter(x=target_node_x, y=target_node_y,
                             mode='markers+text',
                             marker=dict(size=20,
                                       color='#D3D3D3',  # 降低饱和度的浅灰色
                                       symbol='square',
                                       line=dict(width=2, color='#696969')),
                             text=[node.replace('t', 'T') for node in target_nodes],
                             textposition="middle center",
                             textfont=dict(size=8, color='black'),
                             hovertext=target_node_text,
                             hoverinfo='text',
                             name='目标节点'))
    
    # 设置布局 - 去除网格和背景，纯白
    fig.update_layout(
        title=dict(
            text='卫星网络连接关系与目标可见性（交互式）',
            x=0.5,
            font=dict(size=24, color='#2F4F4F')
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=80,l=60,r=60,t=100),
        annotations=[ dict(
            text="💡 操作提示：拖拽平移 | 滚轮缩放 | 悬停查看详情 | 双击重置",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            xanchor='center', yanchor='top',
            font=dict(size=14, color='gray')
        )],
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            visible=False,
            showline=False,
            mirror=False,
            gridcolor='white',
            gridwidth=0
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            visible=False,
            showline=False,
            mirror=False,
            gridcolor='white',
            gridwidth=0
        ),
        plot_bgcolor='white',  # 纯白背景
        paper_bgcolor='white',
        autosize=True,
        dragmode='pan'
    )
    
    # 更新坐标轴以启用所有缩放功能
    fig.update_xaxes(
        fixedrange=False,  # 允许x轴缩放
        scaleanchor=None   # 不锁定比例
    )
    fig.update_yaxes(
        fixedrange=False,  # 允许y轴缩放
        scaleanchor=None   # 不锁定比例
    )
    
    fig.show(config={
        'displayModeBar': True,         # 显示工具栏
        'displaylogo': False,           # 隐藏plotly logo
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': [],   # 不移除任何按钮
        'scrollZoom': True,             # 启用滚轮缩放
        'doubleClick': 'reset+autosize',# 双击重置并自动调整大小
        'showEditInChartStudio': False, # 不显示Chart Studio编辑按钮
        'plotlyServerURL': "https://plot.ly"
    })
    
    # 仅保存静态HTML（无交互高亮）
    html_template = fig.to_html(full_html=True, include_plotlyjs='cdn')
    html_filename = "卫星网络交互式图表.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🎉 交互式图表已显示！")
    print("📋 完整功能列表：")
    print("  🖱️  拖拽鼠标 → 平移视图")
    print("  🔍  滚轮缩放 → 放大/缩小细节")
    print("  📱  悬停节点 → 查看详细信息")
    print("  🔄  双击重置 → 回到初始状态")
    print("  📦  框选缩放 → 选择区域放大")
    print("  🎨  工具栏 → 更多绘图工具")
    print(f"📄  已保存HTML文件：{html_filename}")
    print("🌐  打开HTML文件在浏览器中查看效果")
    
else:
    # 使用matplotlib作为备选方案
    plt.figure(figsize=(20, 20), dpi=100)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制边 - 使用彩色连接，降低透明度
    for edge in sat_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        source_node = edge[0]
        plt.plot([x0, x1], [y0, y1], color=node_edge_color_map[source_node], 
                alpha=0.6, linewidth=2)  # 降低透明度
    
    for edge in target_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        source_node = edge[0]
        plt.plot([x0, x1], [y0, y1], color=node_edge_color_map[source_node], 
                linestyle='--', alpha=0.5, linewidth=1.5)  # 降低透明度
    
    # 绘制卫星节点 - 使用与边匹配的颜色，降低透明度
    for node in sat_nodes_list:
        x, y = pos[node]
        health = G.nodes[node]['health']
        plt.scatter(x, y, s=health*400, c=node_edge_color_map[node], 
                   alpha=0.8, edgecolors='white', linewidth=3)  # 稍微降低透明度
        plt.text(x, y, node.replace('s1', ''), ha='center', va='center', 
                fontsize=10, color='white', weight='bold')
    
    # 绘制目标节点 - 使用更柔和的颜色
    for node in target_nodes:
        x, y = pos[node]
        plt.scatter(x, y, s=400, c='#D3D3D3', marker='s',  # 使用与plotly相同的颜色
                   alpha=0.8, edgecolors='#696969', linewidth=2)
        plt.text(x, y, node.replace('t', 'T'), ha='center', va='center', 
                fontsize=8, color='black', weight='bold')
    
    plt.title('卫星网络连接关系与目标可见性', fontsize=20, fontweight='bold', color='#2F4F4F')
    plt.axis('off')
    plt.tight_layout()
    
    # 设置更柔和的背景色
    plt.gca().set_facecolor('#FAFAFA')
    plt.gcf().patch.set_facecolor('white')
    
    plt.show()
    print("静态图表已显示（matplotlib版本，配色已优化）")

# 显示网络信息
print("卫星网络可视化完成")
print(f"卫星节点数: {len([n for n in G.nodes() if node_types[n] == 'satellite'])}")
print(f"目标节点数: {len([n for n in G.nodes() if node_types[n] == 'target'])}")
print(f"卫星间连接数: {len(sat_edges)}")
print(f"卫星-目标连接数: {len(target_edges)}")

