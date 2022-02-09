import sys
sys.path.append("../")
from src.util.graph_reading import make_figure_path
from src.layout.reduction import tsne
from src.layout.embedding_fr import embedding_fr
import os
import time
from src.util.contants import COLOR_MAP
from src.util.cluster import kmeans
import matplotlib.pyplot as plt
import networkx as nx
from src.embs.metapath2vec.jsonreader import JsonMetaPathGenerator
from src.embs.metapath2vec.genmetapaths import MetaPathGenerator
from src.embs.metapath2vec.metapath2vec import mp2vec


config = {
    # "graph_name": "net_dbis",
    "graph_name": "miserables",
    "seed": 6,
    "te": 0.7,
    "wa": 0.3,
    "we": 1,
    "note": "mp2vec",
    "node_size": 180,
    "edge_width": 0.35,
    "edge_alpha": 0.6,
    "teh": 0.8,
    "epochs": 10,
    "walklen": 10,
    "d": 100
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


def run_metapath2vec_with_layout():
    if config['graph_name'] in ["miserables"]:
        path_gen = JsonMetaPathGenerator()
        path_gen.read_data("./data", "{}.json".format(config['graph_name']))
        if not os.path.exists("./data/{}".format(config['graph_name'])):
            os.mkdir("./data/{}".format(config['graph_name']))
        path_file = "./data/{}/path.txt".format(config['graph_name'])
        type_map = "./data/{}/type_map.txt".format(config['graph_name'])
    else:
        path_gen = MetaPathGenerator()
        path_gen.read_data("./data/net_dbis")
        path_file = "./data/net_dbis/path.txt"
        type_map = "./data/net_dbis/type_map.txt"
    path_gen.generate_random_aca(
        outfilename=path_file,
        numwalks=config['epochs'],
        walklength=config['walklen']
    )
    path_gen.generate_node_type_map(
        path_file=path_file,
        type_map_file=type_map
    )
    start = time.time()
    vectors = mp2vec(
        d=config['d'],
        walk_txt=path_file,
        node_type_txt=type_map
    )
    end = time.time()
    embedding_t = round(start - end, 3)
    print("embedding ends.")
    G = path_gen.generate_nx_graph(vectors)
    cluster = kmeans(vectors, K=7)
    color_list = []
    for node in G.nodes:
        color_list.append(COLOR_MAP[cluster[node]])

    # 绘图
    save_path = make_figure_path(config['graph_name'], note="")
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # TSNE
    print("TSNE")
    start = time.time()
    tsne_pos = tsne(G, vectors)
    end = time.time()
    tsne_t = round(start - end, 3)
    nx.draw_networkx_nodes(
        G, tsne_pos, node_size=config['node_size'], node_color=color_list,
        edgecolors='white', linewidths=0.7)
    plt.savefig(save_path + "/{}-EM-{}s.png".format(
        config['graph_name'].replace("/", "_").replace(".", "_"), embedding_t))
    plt.cla()

    print("FR")
    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
    end = time.time()
    fr_t = round(end - start, 3)
    nx.draw_networkx_nodes(
        G, fr_pos, node_size=config['node_size'], node_color=color_list,
        edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(
        G, fr_pos, width=config['edge_width'], alpha=config['edge_alpha'])
    plt.savefig(save_path + "/{}-FR-{}s.png".format(
        config['graph_name'].replace("/", "_").replace(".", "_"), fr_t))
    plt.cla()

    # EM+FR
    print("EM+FR")
    start = time.time()
    em_fr_pos = embedding_fr(
        G, vectors=vectors, te=config['te'], wa=config['wa'], we=config['we'])
    end = time.time()
    em_fr_t = round(end - start, 3)
    nx.draw_networkx_nodes(
        G, em_fr_pos, node_size=config['node_size'], node_color=color_list,
        edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(
        G, em_fr_pos, width=config['edge_width'], alpha=config['edge_alpha'])
    plt.savefig(save_path + "/{}-EM-FR-{}s.png".format(
        config['graph_name'].replace("/", "_").replace(".", "_"), em_fr_t))
    plt.cla()

    print("EM+FR+TEH")
    start = time.time()
    em_fr_cte_pos = embedding_fr(
        G, vectors=vectors, te=config['te'], wa=config['wa'], we=config['we'], cluster=cluster, teh=config['teh'])
    end = time.time()
    em_fr_cte_t = round(end - start, 3)
    nx.draw_networkx_nodes(
        G, em_fr_cte_pos, node_size=config['node_size'], node_color=color_list,
        edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, em_fr_cte_pos)
    nx.draw_networkx_edges(
        G, em_fr_cte_pos, width=config['edge_width'], alpha=config['edge_alpha'])
    plt.savefig(
        save_path + "/{}-EM-FR-CTE-{}s.png".format(
            config['graph_name'].replace("/", "_").replace(".", "_"), em_fr_cte_t))
    plt.cla()

    if config['graph_name'] in ["miserables"]:
        real_types = path_gen.get_real_types()
        class_map = {}
        count = 0
        real_color_list = []
        if real_types:
            for c in real_types.values():
                if c in class_map.keys():
                    continue
                class_map[c] = count
                count += 1
        for node in G.nodes:
            c = real_types[node]
            c_num = class_map[c]
            real_color_list.append(COLOR_MAP[c_num])
        nx.draw_networkx_nodes(
            G, tsne_pos, node_size=config['node_size'], node_color=real_color_list,
            edgecolors='white', linewidths=0.7)
        plt.savefig(save_path + "/{}-EM-{}s-realcolor.png".format(
            config['graph_name'].replace("/", "_").replace(".", "_"), embedding_t))
        plt.cla()
        nx.draw_networkx_nodes(
            G, fr_pos, node_size=config['node_size'], node_color=real_color_list,
            edgecolors='white', linewidths=0.7)
        nx.draw_networkx_edges(
            G, fr_pos, width=config['edge_width'], alpha=config['edge_alpha'])
        plt.savefig(save_path + "/{}-FR-{}s-realcolor.png".format(
            config['graph_name'].replace("/", "_").replace(".", "_"), fr_t))
        plt.cla()
        nx.draw_networkx_nodes(
            G, em_fr_pos, node_size=config['node_size'], node_color=real_color_list,
            edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_pos)
        nx.draw_networkx_edges(
            G, em_fr_pos, width=config['edge_width'], alpha=config['edge_alpha'])
        plt.savefig(save_path + "/{}-EM-FR-{}s-realcolor.png".format(
            config['graph_name'].replace("/", "_").replace(".", "_"), em_fr_t))
        plt.cla()
        nx.draw_networkx_nodes(
            G, em_fr_cte_pos, node_size=config['node_size'], node_color=real_color_list,
            edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_cte_pos)
        nx.draw_networkx_edges(
            G, em_fr_cte_pos, width=config['edge_width'], alpha=config['edge_alpha'])
        plt.savefig(
            save_path + "/{}-EM-FR-CTE-{}s-realcolor.png".format(
                config['graph_name'].replace("/", "_").replace(".", "_"), em_fr_cte_t))
        plt.cla()
    plt.close()
    print("Done.")
