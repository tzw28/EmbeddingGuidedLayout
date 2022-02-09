import sys
import os
sys.path.append("..")
sys.path.extend(
    [os.path.join(root, name) for root, dirs, _ in os.walk("../")
     for name in dirs]
)
import matplotlib.pyplot as plt
from src.util.cluster import kmeans
from src.util.graph_reading import read_json_graph
from karateclub import MUSAE
from karateclub import NetMF
import networkx as nx



COLOR_MAP = {
    0: "tab:blue",
    1: "tab:brown",
    2: "tab:orange",
    3: "tab:pink",
    4: "tab:green",
    5: "tab:red",
    6: "tab:olive",
    7: "tab:purple",
    8: "tab:cyan",
    9: "limegreen",
    10: "teal",
    11: "tan",
    12: "lavender",
    13: "dodgerblue",
    14: "maroon",
}


def init_fig(fig_size, ax_gap, frame_visible=True, lim=None):
    plt.close()
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = plt.axes([ax_gap, ax_gap, 1 - ax_gap * 2, 1 - ax_gap * 2])
    ax.spines["top"].set_visible(frame_visible)
    ax.spines["right"].set_visible(frame_visible)
    ax.spines["bottom"].set_visible(frame_visible)
    ax.spines["left"].set_visible(frame_visible)
    if lim:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    return fig, ax


g = read_json_graph('data/miserables.json')

# g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)
mapping = {}
for i, node in enumerate(list(g.nodes)):
    mapping[node] = i
g = nx.relabel_nodes(g, mapping)
# model = NetMF(order=3)
model = MUSAE()
model.fit(g)
emb = list(model.get_embedding())

vectors = {}
for i, node in enumerate(list(g.nodes)):
    vectors[node] = emb[i]

pos = nx.fruchterman_reingold_layout(g, seed=17)
colors = kmeans(vectors, K=11)
nodes = g.nodes
color_list = []
for node in nodes:
    color_list.append(COLOR_MAP[colors[node]])
nx.draw_networkx_nodes(
    g, pos, node_size=80, node_color=color_list, edgecolors='white', linewidths=0.1)
nx.draw_networkx_edges(g, pos, width=0.5, alpha=0.3)
plt.show()
