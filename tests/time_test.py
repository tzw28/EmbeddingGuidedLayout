from src.util.graph_reading import (
    init_attributed_graph,
    clean_attributed_graph,
    make_figure_path,
    read_cora_graph,
    read_citeseer_graph,
    read_ppi_graph,
    read_ppt_graph,
    read_miserables_graph,
    read_science_graph,
    read_facebook_graph
)
from src.util.contants import COLOR_MAP
from src.util.cluster import kmeans
from src.layout.embedding_fr import embedding_fr
from src.attr2vec.embedding import attributed_embedding
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import traceback


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


readers = {
    "cora": read_cora_graph,
    "citeseer": read_citeseer_graph,
    "ppi": read_ppi_graph,
    "ppt": read_ppt_graph,
    "miserables": read_miserables_graph,
    "science": read_science_graph,
    "facebook": read_facebook_graph
}


def time_test(
    graph_size,
    d=8,
    walklen=30,
    k=None,
    return_weight=1,
    neighbor_weight=1,
    attribute_weight=1,
    epochs=30,
    seed=6,
    te=0.6, wa=1, we=1, note="",
    teh=0.85,
    node_size=20,
    edge_width=0.5,
    edge_alpha=0.5
):
    # 读取对应图数据，处理自带标签
    if graph_size in list(readers.keys()):
        graph_name = graph_size
        G, _ = readers[graph_size]()
    elif graph_size <= 10000:
        graph_name = "random{}".format(graph_size)
        G = nx.complete_graph(graph_size)
        print("Generate {}".format(graph_name))
    else:
        graph_name = "random_lobster{}".format(graph_size-10000)
        G = nx.random_lobster(graph_size-10000, 0.9, 0.9, seed=12)
        print("Generate {}".format(graph_name))

    # Embedding和聚类、分配颜色
    start = time.time()
    virtual_nodes = init_attributed_graph(G)
    vectors, walks = attributed_embedding(
        G,
        d=d,
        walklen=walklen,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        attribute_weight=attribute_weight,
        epochs=epochs,
        seed=seed,
        virtual_nodes=virtual_nodes,
        graph_name=graph_name,
        get_weights=False)
    clean_attributed_graph(G, vectors)
    cluster = kmeans(vectors, K=k)
    color_list = []
    for node in G.nodes:
        color_list.append(COLOR_MAP[cluster[node]])
    end = time.time()
    emb_time = round(end - start, 3)

    # FR
    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
    end = time.time()
    fr_t = round(end - start, 3)
    # EM+FR(不均截断)
    start = time.time()
    em_fr_cte_pos = embedding_fr(
        G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, teh=teh)
    end = time.time()
    em_fr_cte_t = round(end - start, 3)

    # 绘图
    save_path = make_figure_path(graph_name, note)
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    nx.draw_networkx_nodes(G, fr_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(G, fr_pos, width=edge_width, alpha=edge_alpha)
    plt.savefig(save_path + "/{}-FR-{}s.png".format(graph_name.replace("/",
                "_").replace(".", "_"), fr_t))
    plt.cla()

    nx.draw_networkx_nodes(G, em_fr_cte_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(
        G, em_fr_cte_pos, width=edge_width, alpha=edge_alpha)
    plt.savefig(save_path + "/{}-EM-FR-CTE-{}embs-{}frs-{}s.png".format(graph_name.replace(
        "/", "_").replace(".", "_"), emb_time, em_fr_cte_t, emb_time + em_fr_cte_t))
    plt.close()


def run_time_test():
    print("Time tests Begin...")
    # 读取
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    # 开始测试
    sizes = [10, "miserables", "facebook", "science", "cora", "citeseer",
             10, 10, 100, 200, 500, 800, 1000, 1500, 2000, 3000, 4000,
             5000, 6000, 8000, 10000,
             10010, 10100, 10200, 10500, 10800, 11000, 11500, 12000,
             13000, 14000, 15000, 16000, 18000, 10000]
    for graph_size in sizes:
        try:
            time_test(graph_size=graph_size)
        except Exception as e:
            exstr = traceback.format_exc()
            if graph_size in ["miserables", "facebook", "science", "cora", "citeseer"]:
                graph_name = graph_size
            elif graph_size <= 10000:
                graph_name = "c{}".format(graph_size)
            else:
                graph_name = "lobster{}".format(graph_size-10000)
            make_figure_path(graph_name, "流程测试失败", err_msg=exstr)
            continue
