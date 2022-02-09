from src.util.graph_reading import (
    init_attributed_graph,
    clean_attributed_graph,
    largest_connected_subgraph,
    make_figure_path,
    read_cora_graph,
    read_citeseer_graph,
    read_cornell_graph,
    read_drgraph_resulk,
    read_graphtsne_result,
    read_ppi_graph,
    read_ppt_graph,
    read_miserables_graph,
    read_science_graph,
    read_facebook_graph,
    read_nature150_graph,
    generate_random_graph,
    read_ph_result,
    read_influence_graph,
    read_influence_lcg,
    read_cora_lcg,
    read_citeseer_lcg,
    read_facebook_lcg,
    read_security_graph,
    read_security_lcg,
    read_terrorist_attacks,
)
from src.util.contants import COLOR_MAP
from src.util.cluster import kmeans
from src.layout.embedding_fr import _normalize_positions, embedding_fr
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


readers = {
    "random": generate_random_graph,
    "cora": read_cora_graph,
    "citeseer": read_citeseer_graph,
    "ppi": read_ppi_graph,
    "ppt": read_ppt_graph,
    "miserables": read_miserables_graph,
    "science": read_science_graph,
    "facebook": read_facebook_graph,
    "nature150": read_nature150_graph,
    "influence": read_influence_graph,
    "influence_lcg": read_influence_lcg,
    "cora_lcg": read_cora_lcg,
    "citeseer_lcg": read_citeseer_lcg,
    "facebook_lcg": read_facebook_lcg,
    "security": read_security_graph,
    "security_lcg": read_security_lcg,
    "terrorist_attacks": read_terrorist_attacks,
    "cornell": read_cornell_graph,
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


def auto_click(agg, fig, n=0, click=None):
    # 随机点开两个聚类
    agg_n = len(list(agg.ag_G.nodes))
    rand_list = []
    for i in range(n):
        if len(rand_list) > agg_n:
            break
        rand_sel = random.randint(0, agg_n - 1)
        while rand_sel in rand_list:
            rand_sel = random.randint(0, agg_n - 1)
        sel_node = list(agg.agg_size.keys())[rand_sel]
        agg.select(sel_node, fig)
        rand_list.append(rand_sel)


def overall_tests(
    graph_name,
    d=8,
    walklen=30,
    k=None,
    return_weight=1,
    neighbor_weight=1,
    attribute_weight=1,
    epochs=30,
    seed=6,
    te=0.6, wa=1, we=1, note="",
    tel=0.5,
    teh=0.85,
    node_size=20,
    edge_width=0.5,
    edge_alpha=0.5,
    agg_size_min=40,
    agg_size_max=40,
    agg_width_min=0.2,
    agg_width_max=0.2,
    agg_alpha=1,
    run_evaluate=False
):
    # 读取对应图数据，处理自带标签
    G, node_class = readers[graph_name]()
    class_map = {}  # 节点标签映射
    count = 0       # 节点标签类型数目
    exist_labels = False
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

    # DRGraph
    if True:
        drgraph_pos = read_drgraph_resulk(graph_name)
    # GraphTSNE
    if graph_name in ["cora", "citeseer", "cornell"]:
        graphtsne_pos = read_graphtsne_result(graph_name)
        # _normalize_positions(graphtsne_pos)

    # Embedding和聚类、分配颜色
    color_list = []
    real_color_list = []
    # if graph_name == "facebook":
    if True:
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
        local_vectors, _ = attributed_embedding(
            G,
            d=d,
            walklen=walklen,
            return_weight=1,
            neighbor_weight=10,
            attribute_weight=1,
            epochs=epochs,
            seed=seed,
            virtual_nodes=[],
            graph_name=graph_name,
            get_weights=False)
        global_vectors, _ = attributed_embedding(
            G,
            d=d,
            walklen=walklen,
            return_weight=10,
            neighbor_weight=1,
            attribute_weight=1,
            epochs=epochs,
            seed=seed,
            virtual_nodes=[],
            graph_name=graph_name,
            get_weights=False)
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

    # 绘图
    save_path = make_figure_path(graph_name, note)
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # DRGraph
    if graph_name in ["facebook", "facebook_lcg", "influence", "influence_lcg"]:
        real_color_list = color_list
    else:
        real_color_list = []
        for node in G.nodes:
            c = node_class[node]
            c_num = class_map[c]
            real_color_list.append(COLOR_MAP[c_num])
    pos_dict = {}
    if True:
        pos_dict["drgraph"] = drgraph_pos
        nx.draw_networkx_nodes(G, drgraph_pos, node_size=node_size,
                               node_color=real_color_list, edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_pos)
        nx.draw_networkx_edges(
            G, drgraph_pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(
            save_path + "/{}-drgraph.png".format(graph_name.replace("/", "_").replace(".", "_")))
        plt.cla()

    # GraphTSNE
    if graph_name in ["cora", "citeseer", "cornell"]:
        pos_dict["graphtsne"] = graphtsne_pos
        nx.draw_networkx_nodes(G, graphtsne_pos, node_size=node_size,
                               node_color=real_color_list, edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_pos)
        nx.draw_networkx_edges(
            G, graphtsne_pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(
            save_path + "/{}-graphtsne.png".format(graph_name.replace("/", "_").replace(".", "_")))
        plt.cla()
    plt.close()
    if False:
        ev = LayoutEvaluator(G, pos_dict, cluster)
        ev.run()
        ev.save_json_result(save_path, graph_name)

    weights = graph_tf_idf(
        G, cluster, walks,
        d=d,
        walklen=walklen,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        attribute_weight=attribute_weight,
        epochs=epochs,
        seed=seed,
        graph_name=graph_name)

    # TSNE
    start = time.time()
    tsne_pos = tsne(G, vectors)
    end = time.time()
    tsne_t = round(start - end, 3)
    # FR
    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
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
        G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=tel, teh=teh)
    end = time.time()
    em_fr_cte_t = round(end - start, 3)
    extra_pos = {}
    if False:
        extra_pos["ex_tsne"] = tsne_pos
        extra_pos["ex_sim_mat"] = embedding_fr(
            G, vectors=vectors, te=te, wa=0, we=we, cluster=cluster, tel=0, teh=0)
        extra_pos["ex_sim_te_0.4"] = embedding_fr(
            G, vectors=vectors, te=te, wa=0, we=we, cluster=cluster, tel=0.4, teh=0.4)
        extra_pos["ex_EA_te_0.4"] = embedding_fr(
            G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=0.4, teh=0.4)
        extra_pos["ex_sim_te_0.5"] = embedding_fr(
            G, vectors=vectors, te=te, wa=0, we=we, cluster=cluster, tel=0.5, teh=0.5)
        extra_pos["ex_EA_te_0.5"] = embedding_fr(
            G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=0.5, teh=0.5)
        extra_pos["ex_sim_te_0.6"] = embedding_fr(
            G, vectors=vectors, te=te, wa=0, we=we, cluster=cluster, tel=0.6, teh=0.6)
        extra_pos["ex_EA_te_0.6"] = embedding_fr(
            G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=0.6, teh=0.6)
        extra_pos["ex_sim_te_0.7"] = embedding_fr(
            G, vectors=vectors, te=te, wa=0, we=we, cluster=cluster, tel=0.7, teh=0.7)
        extra_pos["ex_EA_te_0.7"] = embedding_fr(
            G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=0.7, teh=0.7)
        extra_pos["ex_EA_telh"] = embedding_fr(
            G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster, tel=tel, teh=teh)

    # PH
    try:
        ph_pos = read_ph_result(graph_name)
    except:
        ph_pos = None

    # 绘图
    # save_path = make_figure_path(graph_name, note)
    pos_dict["FR"] = fr_pos
    pos_dict["EGL"] = em_fr_cte_pos
    if ph_pos:
        pos_dict["PH"] = ph_pos
    # pos_dict = {
    #     "FR": fr_pos,
    #     "EM-FR-CTE": em_fr_cte_pos,
    #     # "PH": ph_pos
    # }
    if run_evaluate:
        ev = LayoutEvaluator(G, pos_dict, cluster)
        ev.run()
        ev.save_json_result(save_path, graph_name)

    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # TSNE
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-{}".format(
        "EM", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed, "label" if exist_labels else "cluster")
    # plt.title(fig_title)
    nx.draw_networkx_nodes(G, tsne_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    plt.savefig(save_path + "/{}-EM-{}s.png".format(graph_name.replace("/",
                "_").replace(".", "_"), embedding_t))
    plt.cla()
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-{}".format(
        "FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed, "label" if exist_labels else "cluster")
    # plt.title(fig_title)
    nx.draw_networkx_nodes(G, fr_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, fr_pos)
    nx.draw_networkx_edges(G, fr_pos, width=edge_width, alpha=edge_alpha)
    plt.savefig(save_path + "/{}-FR-{}s.png".format(graph_name.replace("/",
                "_").replace(".", "_"), fr_t))
    plt.cla()
    # EM+FR
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-wa{}-we{}-te{}-{}".format(
        "EM+FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed, wa, we, te, "label" if exist_labels else "cluster")
    # plt.title(fig_title)
    nx.draw_networkx_nodes(G, em_fr_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, em_fr_pos)
    nx.draw_networkx_edges(G, em_fr_pos, width=edge_width, alpha=edge_alpha)
    plt.savefig(save_path + "/{}-EM-FR-{}s.png".format(
        graph_name.replace("/", "_").replace(".", "_"), em_fr_t))
    plt.cla()
    for name, pos in extra_pos.items():
        nx.draw_networkx_nodes(G, pos, node_size=node_size,
                               node_color=color_list, edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_pos)
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(save_path + "/{}.png".format(name))
        plt.cla()
    # PH
    if ph_pos:
        LCG = largest_connected_subgraph(G)
        lcg_degrees = []
        lcg_nodes = list(LCG.nodes)
        for node in lcg_nodes:
            lcg_degrees.append(LCG.degree[node])
        if len(list(LCG.nodes)) < 100:
            lcg_node_size = [size_value / 2 + 7 *
                             degree for degree in lcg_degrees]
        else:
            lcg_node_size = [size_value for degree in lcg_degrees]
        lcg_color_list = []
        if class_map:
            for node in LCG.nodes:
                c = node_class[node]
                c_num = class_map[c]
                lcg_color_list.append(COLOR_MAP[c_num])
        else:
            for node in LCG.nodes:
                lcg_color_list.append(COLOR_MAP[cluster[node]])
        nx.draw_networkx_nodes(LCG, ph_pos, node_size=lcg_node_size,
                               node_color=lcg_color_list, edgecolors='white', linewidths=0.7)
        # nx.draw_networkx_labels(G, em_fr_pos)
        nx.draw_networkx_edges(LCG, ph_pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(save_path + "/{}-PH-{}s.png".format(
            graph_name.replace("/", "_").replace(".", "_"), em_fr_t))
        plt.cla()
    # EM+FR
    if class_map and count:
        nx.draw_networkx_nodes(G, fr_pos, node_size=node_size,
                               node_color=real_color_list, edgecolors='white', linewidths=0.7)
        nx.draw_networkx_edges(G, fr_pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(save_path + "/{}-FR-RealColor-{}s.png".format(
            graph_name.replace("/", "_").replace(".", "_"), em_fr_t))
        plt.cla()
        nx.draw_networkx_nodes(G, em_fr_cte_pos, node_size=node_size,
                               node_color=real_color_list, edgecolors='white', linewidths=0.7)
        nx.draw_networkx_edges(
            G, em_fr_cte_pos, width=edge_width, alpha=edge_alpha)
        plt.savefig(save_path + "/{}-EM-FR-CTE-RealColor-{}s.png".format(
            graph_name.replace("/", "_").replace(".", "_"), em_fr_t))
        plt.cla()

    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-wa{}-we{}-te{}-{}".format(
        "EM+FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed, wa, we, te, "label" if exist_labels else "cluster")
    # plt.title(fig_title)
    nx.draw_networkx_nodes(G, em_fr_cte_pos, node_size=node_size,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, em_fr_cte_pos)
    nx.draw_networkx_edges(
        G, em_fr_cte_pos, width=edge_width, alpha=edge_alpha)
    plt.savefig(save_path + "/{}-EM-FR-CTE-{}s.png".format(
        graph_name.replace("/", "_").replace(".", "_"), em_fr_cte_t))
    plt.close()

    # EM+FR+AGG
    # for i in range(1):
    #     start = time.time()
    #     fig, ax = init_fig(10, 0.05, frame_visible=False)
    #     add_group_attr(G, cluster)
    #     agg = GraphAggregator(
    #         G,
    #         em_fr_cte_pos,
    #         group_attribute="ag_group",
    #         fig_size=10,
    #         ax_gap=0.05,
    #         is_curved=True,
    #         attr_vectors=vectors,
    #         local_vectors=local_vectors,
    #         global_vectors=global_vectors,
    #         weights=weights
    #     )
    #     agg.generate_aggregations()
    #     agg.draw_aggregations(
    #         size_min=agg_size_min,
    #         size_max=agg_size_max,
    #         width_min=agg_width_min,
    #         width_max=agg_width_max,
    #         agg_alpha=agg_alpha,
    #     )
    #     agg.set_events(fig)
    #     end = time.time()
    #     t = round(end - start, 3)
    #     if i == 0:
    #         plt.savefig(save_path + "/{}-EM+FR+AGG-{}s-{}.png".format(graph_name.replace("/", "_").replace(".", "_"), t, i))
    #     plt.show()
    #     # agg.select(i, fig)
    #     # plt.savefig(save_path + "/{}-EM+FR+AGG+Clicked-{}s-{}.png".format(graph_name.replace("/", "_").replace(".", "_"), t, i))
    #     plt.close()
    # start = time.time()
    # fig, ax = init_fig(10, 0.05, frame_visible=False, lim=(0, 1))
    # draw_embedding_fr(
    #     graph_file=graph_name,
    #     default_pos=em_fr_cte_pos,
    #     attribute_weight=attribute_weight,
    #     neighbor_weight=neighbor_weight,
    #     return_weight=return_weight,
    #     seed=seed,
    #     color=cluster,
    #     size_list=node_size,
    #     width=edge_width,
    #     edge_alpha=edge_alpha,
    #     te=te,
    #     wa=wa,
    #     we=we,
    #     teh=teh,
    #     fig=fig,
    #     ax=ax,
    #     ax_gap=0.05,
    #     fig_size=10,
    #     click_on=True,
    # )
    # end = time.time()
    # t = round(end - start, 3)
    # plt.savefig(
    #     save_path
    #     + "/{}-EM+FR+Interaction-{}s.png".format(graph_name.replace("/", "_").replace(".", "_"), t)
    # )
    # plt.show()
    # plt.close()


def run_overall_tests():
    """
    读取配置，对多个图进行流程测试
    """
    print("Tests Begin...")
    # 读取
    test_graph_file = "overall_graphs.json"
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    with open(test_graph_file, "r", encoding="utf-8") as f:
        string = f.read()
        test_graphs = json.loads(string)

    # 开始测试
    for graph_file in test_graphs.keys():
        # 执行测试
        # if graph_file not in ["miserables", "facebook", "cornell", "citeseer"]:
        #     continue
        setting = test_graphs[graph_file]
        try:
            overall_tests(
                graph_file,
                d=setting["d"],
                walklen=setting["walklen"],
                k=setting["k"],
                attribute_weight=setting["attribute_weight"],
                neighbor_weight=setting["neighbor_weight"],
                return_weight=setting["return_weight"],
                epochs=setting["epochs"],
                seed=setting["seed"],
                te=setting["te"],
                wa=setting["wa"],
                we=setting["we"],
                note=setting["note"],
                node_size=setting["node_size"],
                edge_width=setting["edge_width"],
                edge_alpha=setting["edge_alpha"],
                agg_size_min=setting["agg_size_min"],
                agg_size_max=setting["agg_size_max"],
                agg_width_min=setting["agg_width_min"],
                agg_width_max=setting["agg_width_max"],
                agg_alpha=setting["agg_alpha"],
                tel=setting["tel"],
                teh=setting["teh"],
                run_evaluate=False
            )
        except Exception as e:
            exstr = traceback.format_exc()
            print(exstr)
            make_figure_path(graph_file, "failed", err_msg=exstr)
            continue


def run_wa_we_test():
    """
    读取配置，对多个图进行流程测试
    """
    print("Tests Begin...")
    # 读取
    test_graph_file = "overall_graphs.json"
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    with open(test_graph_file, "r", encoding="utf-8") as f:
        string = f.read()
        test_graphs = json.loads(string)
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tel = [0.4, 0.5, 0.6, 0.7]
    teh = [0.5, 0.6, 0.7, 0.8]
    # 开始测试
    for graph_file in test_graphs.keys():
        # 执行测试
        # if graph_file not in ["cornell", "miserables", "science"]:
        #     continue
        setting = test_graphs[graph_file]
        for wa in w:
            we = 1 - wa
            for l in tel:
                for h in teh:
                    if l >= h:
                        continue
                    try:
                        overall_tests(
                            graph_file,
                            d=setting["d"],
                            walklen=setting["walklen"],
                            k=setting["k"],
                            attribute_weight=setting["attribute_weight"],
                            neighbor_weight=setting["neighbor_weight"],
                            return_weight=setting["return_weight"],
                            epochs=setting["epochs"],
                            seed=setting["seed"],
                            te=setting["te"],
                            tel=l,
                            wa=wa,
                            we=we,
                            note="wa{}-we{}-tel{}-teh{}".format(wa, we, l, h),
                            node_size=setting["node_size"],
                            edge_width=setting["edge_width"],
                            edge_alpha=setting["edge_alpha"],
                            agg_size_min=setting["agg_size_min"],
                            agg_size_max=setting["agg_size_max"],
                            agg_width_min=setting["agg_width_min"],
                            agg_width_max=setting["agg_width_max"],
                            agg_alpha=setting["agg_alpha"],
                            teh=h
                        )
                    except Exception as e:
                        exstr = traceback.format_exc()
                        print(exstr)
                        make_figure_path(graph_file, "failed", err_msg=exstr)
                        continue


def run_pqr_test():
    """
    读取配置，对多个图进行流程测试
    """
    print("Tests Begin...")
    # 读取
    test_graph_file = "overall_graphs.json"
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    with open(test_graph_file, "r", encoding="utf-8") as f:
        string = f.read()
        test_graphs = json.loads(string)

    # 开始测试
    for graph_file in test_graphs.keys():
        # 执行测试
        if graph_file not in ["miserables"]:
            continue
        setting = test_graphs[graph_file]
        pqr_list = [0.2, 0.5, 1, 1.5, 2.0, 2.5, 4.0]
        for p in pqr_list:
            for q in pqr_list:
                for r in pqr_list:
                    overall_tests(
                        graph_file,
                        d=setting["d"],
                        walklen=setting["walklen"],
                        k=setting["k"],
                        attribute_weight=1/r,
                        neighbor_weight=1/q,
                        return_weight=1/p,
                        epochs=setting["epochs"],
                        seed=setting["seed"],
                        te=setting["te"],
                        wa=setting["wa"],
                        we=setting["we"],
                        note="p{}-q{}-r{}".format(p, q, r),
                        node_size=setting["node_size"],
                        edge_width=setting["edge_width"],
                        edge_alpha=setting["edge_alpha"],
                        agg_size_min=setting["agg_size_min"],
                        agg_size_max=setting["agg_size_max"],
                        agg_width_min=setting["agg_width_min"],
                        agg_width_max=setting["agg_width_max"],
                        agg_alpha=setting["agg_alpha"],
                        teh=setting["teh"]
                    )
