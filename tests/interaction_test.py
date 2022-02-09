from src.util.graph_reading import (
    init_attributed_graph,
    clean_attributed_graph,
    make_figure_path,
    read_miserables_graph,
    generate_random_graph,
)
from src.util.contants import COLOR_MAP
from src.util.cluster import kmeans
from src.layout.embedding_fr import embedding_fr, _normalize_positions
from src.layout.evaluator import LayoutEvaluator
from src.layout.reduction import tsne
from src.attr2vec.embedding import attributed_embedding
from src.util.plot_drawing import draw_embedding_fr
from src.util.key_words import graph_tf_idf
from src.aggregation.graph_aggregation import (
    add_group_attr,
    GraphAggregator
)
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import json
import random
import traceback
from src.layout.my_fr import run_my_fr


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


def run_interaction_test(graph_name="miserables",
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
                         edge_alpha=0.5,
                         agg_size_min=40,
                         agg_size_max=40,
                         agg_width_min=0.2,
                         agg_width_max=0.2,
                         agg_alpha=1):
    # 读取对应图数据，处理自带标签
    G, node_class = read_miserables_graph()
    class_map = {}  # 节点标签映射
    count = 0       # 节点标签类型数目
    if node_class is not None:
        exist_labels = True
        for c in node_class.values():
            if c in class_map.keys():
                continue
            class_map[c] = count
            count += 1
    # FR
    node_size = node_size
    size_value = node_size
    node_size = []
    nodes = G.nodes
    degrees = []
    for node in nodes:
        degrees.append(G.degree[node])
    if len(list(G.nodes)) < 100:
        node_size = [size_value / 2 + 7 * degree for degree in degrees]
    else:
        node_size = [size_value for degree in degrees]

    # Embedding和聚类、分配颜色
    print("Embedding starts.")
    start = time.time()
    if graph_name in []:
        virtual_nodes = init_attributed_graph(G, skip=True)
    else:
        virtual_nodes = init_attributed_graph(G)
    # vectors, weights = attributed_embedding(
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
        get_weights=True)
    clean_attributed_graph(G, vectors)
    end = time.time()
    embedding_t = end - start
    color_list = []
    real_color_list = []
    if class_map and count:
        for node in G.nodes:
            c = node_class[node]
            c_num = class_map[c]
            real_color_list.append(COLOR_MAP[c_num])
        cluster = kmeans(vectors, K=k)
    elif k:
        print("k=", k)
        cluster = kmeans(vectors, K=k)
    else:
        km_start = time.time()
        cluster = kmeans(vectors, K=None)
        km_end = time.time()
        print("K-Means time: {}s".format(round(km_end - km_start, 3)))
    for node in G.nodes:
        color_list.append(COLOR_MAP[cluster[node]])

    # FR
    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
    fr_pos = _normalize_positions(fr_pos)
    end = time.time()
    fr_t = round(end - start, 3)
    # My FR
    start = time.time()
    fr_pos = run_my_fr(G)
    # fr_pos = good_fruchterman_reingold_layout(G)
    _normalize_positions(fr_pos)
    end = time.time()
    fr_t = round(end - start, 3)
    # EM+FR
    start = time.time()
    em_fr_pos = embedding_fr(G, vectors=vectors, te=te, wa=wa, we=we)
    end = time.time()
    em_fr_t = round(end - start, 3)
    # EM+FR(不均截断)
    start = time.time()
    em_fr_cte_pos = embedding_fr(
        G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, teh=teh)
    end = time.time()
    em_fr_cte_t = round(end - start, 3)

    # 绘图
    # save_path = make_figure_path(graph_name, note)
    pos_dict = {
        "FR": fr_pos,
        "EM-FR-CTE": em_fr_cte_pos
    }

    start = time.time()
    fig, ax = init_fig(10, 0.05, frame_visible=False, lim=(0, 1))
    draw_embedding_fr(
        graph_file=graph_name,
        default_pos=fr_pos,
        attribute_weight=attribute_weight,
        neighbor_weight=neighbor_weight,
        return_weight=return_weight,
        seed=seed,
        color=cluster,
        size_list=node_size,
        width=edge_width,
        edge_alpha=edge_alpha,
        te=te,
        wa=wa,
        we=we,
        teh=teh,
        fig=fig,
        ax=ax,
        ax_gap=0.05,
        fig_size=10,
        click_on=True,
    )
    end = time.time()
    t = round(end - start, 3)
    # plt.savefig(
    #     save_path
    #     + "/{}-EM+FR+Interaction-{}s.png".format(graph_name.replace("/", "_").replace(".", "_"), t))
    plt.show()
    plt.close()
