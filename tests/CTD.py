from src.util.graph_reading import read_chemical_disease_graph
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
import time
import os

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


def draw_ctd_cure_graph(
    d=8,
    walklen=30,
    return_weight=1,
    neighbor_weight=4,
    attribute_weight=0.5,
    epochs=30,
    k=7,
    seed=6,
    color=None,
    size_min=8, size_max=8,
    width=0.5, edge_alpha=0.7,
    te=0.6, wa=1, we=1, note=""
):
    graph_file = "CTD"
    G = read_chemical_disease_graph("data/")
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    print("Embedding starts.")
    start = time.time()
    virtual_nodes = init_attributed_graph(G)
    vectors, weights = attributed_embedding(
        G,
        d=d,
        walklen=walklen,
        return_weight=1/return_weight,
        neighbor_weight=1/neighbor_weight,
        attribute_weight=1/attribute_weight,
        epochs=epochs,
        seed=seed,
        virtual_nodes=virtual_nodes,
        graph_name=graph_file,
        get_weights=False)
    clean_attributed_graph(G, vectors)

    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    time_str = time.strftime("%H-%M-%S", time.localtime(time.time()))
    save_path = fig_path + "/" + graph_file.replace(".", "_")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += "/" + time_str + "-" + note
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cluster_color = kmeans(vectors, K=10)
    cluster_color_list = []
    for node in G.nodes:
        cluster_color_list.append(COLOR_MAP[cluster_color[node]])
    tsne_pos = tsne(G, vectors)
    end = time.time()
    em_t = round(end - start, 3)
    plt.figure(figsize=(10, 10))
    plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}".format(
        "EM", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    )
    plt.title(fig_title)
    nx.draw_networkx_nodes(G, tsne_pos, node_size=15,
                           node_color=cluster_color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, tsne_pos, labels=labels, font_size=10)
    plt.savefig(
        save_path
        + "/{}-EM-{}s.png".format(graph_file.replace("/",
                                  "_").replace(".", "_"), em_t)
    )
    plt.close()

    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
    end = time.time()
    fr_t = round(end - start, 3)
    plt.figure(figsize=(10, 10))
    plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}".format(
        "FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    )
    plt.title(fig_title)
    nx.draw_networkx_nodes(G, fr_pos, node_size=15,
                           node_color=cluster_color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, fr_pos, labels=labels, font_size=10)
    nx.draw_networkx_edges(G, fr_pos, width=0.5, alpha=0.3)
    plt.savefig(
        save_path
        + "/{}-FR-{}s.png".format(graph_file.replace("/",
                                  "_").replace(".", "_"), fr_t)
    )
    plt.close()
