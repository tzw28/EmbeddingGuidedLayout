import networkx as nx
import matplotlib.pyplot as plt
from src.util.graph_reading import (
    init_attributed_graph,
    clean_attributed_graph
)
from src.util.normalize import normalize_list
from src.util.cluster import kmeans
from src.layout.embedding_fr import embedding_fr
from src.layout.reduction import tsne
from src.attr2vec.embedding import attributed_embedding

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
    12: "salmon",
    13: "dodgerblue",
    14: "maroon",
}


def facebook_reading_test():
    feature_name_file = "data/facebook/0.featnames"
    node_feature_file = "data/facebook/0.feat"
    edge_file = "data/facebook/0.edges"

    feature_list = []
    with open(feature_name_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            feature_content = strs[-1].strip()
            feature_name = "_".join(strs[1].split(";")[:-1])
            feature = (feature_name, "f" + feature_content)
            feature_list.append(feature)
    # print(feature_list)
    G = nx.Graph()
    with open(node_feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            node_id = strs[0]
            G.add_node(node_id)
            for i, f in enumerate(strs[1:]):
                if int(f) == 0:
                    continue
                feature = feature_list[i]
                G.nodes[node_id][feature[0]] = feature[1]
    with open(edge_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0]
            t = strs[1].strip()
            G.add_edge(s, t)
    return G


def draw_facebook_embedding_fr(
        d=8,
        walklen=30,
        return_weight=1,
        neighbor_weight=1,
        attribute_weight=1,
        epochs=30,
        k=7,
        seed=None,
        color=None,
        size_min=8, size_max=8,
        width=0.5, edge_alpha=0.7,
        te=0.6, wa=1, we=1,):
    G = facebook_reading_test()
    virtual_nodes = init_attributed_graph(G)
    vectors, weights = attributed_embedding(
        G,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        attribute_weight=attribute_weight,
        epochs=epochs,
        seed=seed,
        virtual_nodes=virtual_nodes,
        graph_name="facebook0")
    clean_attributed_graph(G, vectors)
    # pos = tsne(G, vectors)
    # pos = embedding_fr(G, vectors=vectors, te=te, wa=wa, we=we)
    pos = nx.fruchterman_reingold_layout(G, seed=17)
    color = kmeans(vectors, K=k)
    color_list = []
    for node in G.nodes:
        color_list.append(COLOR_MAP[color[node]])
    nx.draw_networkx_nodes(
        G, pos, node_size=50, edgecolors='white', node_color=color_list, linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    plt.show()
