import sys
import os
sys.path.append("..")
sys.path.extend(
    [os.path.join(root, name) for root, dirs, _ in os.walk("../")
     for name in dirs]
)
import random
import matplotlib.pyplot as plt
from src.attr2vec.embedding import attributed_embedding
from src.aggregation.graph_aggregation import add_group_attr, GraphAggregator
from src.util.plot_drawing import draw_attr2vec_tsne, draw_nx_fr, draw_embedding_fr
import json
import time


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


def auto_click(agg, fig, n=0):
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


def process_tests(
    graph_file,  # 图文件名
    save_path,  # 结果保存路径
    attribute_weight=20,  # 1 / r
    neighbor_weight=1,  # 1 / q
    return_weight=1,  # 1 / p
    cluster_method="km",  # 聚类方法
    k=9,  # 聚类数
    seed=6,  # 随机数种子
    te=0.6,  # Embedding权重矩阵截断比例
    wa=1,  # 原始邻接矩阵权重
    we=1,  # Embedding邻接矩阵权重
    fr_size_min=8,  # FR布局中节点的最小大小
    fr_size_max=4,  # FR布局中节点的最大大小
    fr_width=0.2,  # FR布局中边宽度
    key_node_lim=15,  # Embedding指导的FR的参数
    key_node_method="degree",  # 选择关键节点的依据
    agg_size_min=40,  # 聚合中节点的最小大小
    agg_size_max=40,  # 聚合中节点的最大大小
    agg_width_min=0.2,  # 聚合中边的最小大小
    agg_width_max=0.2,  # 聚合中边的最大大小
    agg_alpha=1,  # 聚合后，点开节点后，白色背景的透明度
    fig_size=10,  # 画布大小
    ax_gap=0.05,  # 子图左边缘、下边缘的比例
    edge_alpha=0.7,  # 边透明度,
    random_click=2,  # 最多自动点开多少个点
    itrac_mode=False,  # 交互模式
):
    start = time.time()
    print("Data: ", graph_file)

    # 第一图：EM + tSNE
    init_fig(fig_size, ax_gap, frame_visible=False)
    # plt.title = "Attr2vec(aw={})+K-Means(k={})+t-SNE".format(attribute_weight, k)
    vectors, weights, G, pos, colors = draw_attr2vec_tsne(
        graph_file=graph_file,
        attribute_weight=attribute_weight,
        neighbor_weight=neighbor_weight,
        return_weight=return_weight,
        cluster_method=cluster_method,
        k=k,
        seed=seed,
        size_min=fr_size_min,
        size_max=fr_size_max,
        width=0,
    )
    end = time.time()
    t = round(end - start, 3)
    plt.savefig(
        save_path
        + "/{}-EM-{}s-aw{}-nw{}-rw{}.png".format(
            graph_file.replace("/", "_").replace(".", "_"),
            t,
            attribute_weight,
            neighbor_weight,
            return_weight)
    )
    plt.close()
    print("Time: ", t)

    # 第二图：FR
    print("FR")
    start = time.time()
    init_fig(fig_size, ax_gap, frame_visible=False)
    draw_nx_fr(
        graph_file=graph_file,
        color=colors,
        size_min=fr_size_min,
        size_max=fr_size_max,
        width=fr_width,
        edge_alpha=edge_alpha,
    )
    end = time.time()
    t = round(end - start, 3)
    plt.savefig(
        save_path
        + "/{}-FR-{}s.png".format(graph_file.replace("/",
                                  "_").replace(".", "_"), t)
    )
    plt.close()
    print("Time: ", t)

    # 第三图：EM + AGG
    print("EM + AGG")
    start = time.time()
    init_fig(fig_size, ax_gap, frame_visible=False)
    # plt.title = "Attr2vec(aw={})+K-Means(k={})+t-SNE+aggregation".format(
    #     attribute_weight, k
    # )
    add_group_attr(G, colors)
    agg = GraphAggregator(
        G, pos, group_attribute="ag_group", key_node_lim=key_node_lim)
    agg.generate_aggregations()
    agg.draw_aggregations(
        size_min=agg_size_min,
        size_max=agg_size_max,
        width_min=agg_width_min,
        width_max=agg_width_max,
        agg_alpha=agg_alpha,
    )
    end = time.time()
    t = round(end - start, 3)
    plt.savefig(
        save_path
        + "/{}-EM+AGG-{}s.png".format(graph_file.replace("/", "_").replace(".", "_"), t)
    )
    plt.close()
    print("Time: ", t)

    # 第四图：EM + FR
    print("EM + FR")
    start = time.time()
    fig, ax = init_fig(fig_size, ax_gap, frame_visible=False, lim=(0, 1))
    # plt.title = "Attr2vec(aw={})+K-Means(k={})+F-R".format(attribute_weight, k)
    G, pos = draw_embedding_fr(
        graph_file=graph_file,
        attribute_weight=attribute_weight,
        neighbor_weight=neighbor_weight,
        return_weight=return_weight,
        seed=seed,
        color=colors,
        size_min=fr_size_min,
        size_max=fr_size_max,
        width=fr_width,
        edge_alpha=edge_alpha,
        te=te,
        wa=wa,
        we=we,
        fig=fig,
        ax=ax,
        ax_gap=ax_gap,
        fig_size=fig_size,
        click_on=True,
    )
    end = time.time()
    t = round(end - start, 3)
    plt.savefig(
        save_path
        + "/{}-EM+FR-{}s.png".format(graph_file.replace("/", "_").replace(".", "_"), t)
    )
    if itrac_mode:
        plt.show()
    plt.close()
    print("Time: ", t)

    # 第五图：EM + FR + AGG + CUR
    print("EM + FR + AGG + CUR")
    start = time.time()
    fig, ax = init_fig(fig_size, ax_gap, frame_visible=False, lim=(0, 1))
    global_vectors, _ = attributed_embedding(
        G,
        seed=seed,
        return_weight=attribute_weight,
        virtual_nodes=[],
        graph_name=graph_file)
    local_vectors, _ = attributed_embedding(
        G,
        seed=seed,
        neighbor_weight=attribute_weight,
        virtual_nodes=[],
        graph_name=graph_file)
    add_group_attr(G, colors)
    agg = GraphAggregator(
        G,
        pos,
        group_attribute="ag_group",
        fig_size=fig_size,
        ax_gap=ax_gap,
        attr_vectors=vectors,
        local_vectors=local_vectors,
        global_vectors=global_vectors,
        weights=weights,
        key_node_lim=key_node_lim,
        key_node_method=key_node_method,
    )
    agg.generate_aggregations()
    agg.draw_aggregations(
        size_min=agg_size_min,
        size_max=agg_size_max,
        width_min=agg_width_min,
        width_max=agg_width_max,
        agg_alpha=agg_alpha,
    )
    agg.set_events(fig)
    auto_click(agg, fig, n=random_click)
    end = time.time()
    t = round(end - start, 3)
    print("Time: ", t)
    plt.savefig(
        save_path
        + "/{}-EM+FR+AGG+HCI+CUR-{}s.png".format(
            graph_file.replace("/", "_").replace(".", "_"), t
        )
    )
    if itrac_mode:
        plt.show()
    plt.close()

    # 第六图：EM + FR + AGG
    print("EM + FR + AGG")
    start = time.time()
    fig, ax = init_fig(fig_size, ax_gap, frame_visible=False, lim=(0, 1))
    add_group_attr(G, colors)
    agg = GraphAggregator(
        G,
        pos,
        group_attribute="ag_group",
        fig_size=fig_size,
        ax_gap=ax_gap,
        is_curved=False,
        attr_vectors=vectors,
        local_vectors=local_vectors,
        global_vectors=global_vectors,
        weights=weights,
        key_node_lim=key_node_lim,
        key_node_method=key_node_method,
    )
    agg.generate_aggregations()
    agg.draw_aggregations(
        size_min=agg_size_min,
        size_max=agg_size_max,
        width_min=agg_width_min,
        width_max=agg_width_max,
        agg_alpha=agg_alpha,
    )
    agg.set_events(fig)
    auto_click(agg, fig, n=random_click)
    end = time.time()
    t = round(end - start, 3)
    plt.savefig(
        save_path
        + "/{}-EM+FR+AGG+HCI-{}s.png".format(
            graph_file.replace("/", "_").replace(".", "_"), t
        )
    )
    # plt.show()
    plt.close()
    print("Time: ", t)


def run_process_tests():
    """
    读取配置，对多个图进行流程测试
    """
    print("TEST BEGIN...")
    # 读取
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_path + "/..")
    test_graph_file = "test_graphs.json"
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    with open(test_graph_file, "r", encoding="utf-8") as f:
        string = f.read()
        test_graphs = json.loads(string)

    # 开始测试
    for graph_file in test_graphs.keys():
        setting = test_graphs[graph_file]
        file_path = "data/" + graph_file
        time_str = time.strftime("%H-%M-%S", time.localtime(time.time()))
        save_path = fig_path + "/" + graph_file.replace(".", "_")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += "/" + time_str
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 执行测试
        process_tests(
            file_path,
            save_path,
            # Emebedding
            attribute_weight=setting["attribute_weight"],
            neighbor_weight=setting["neighbor_weight"],
            return_weight=setting["return_weight"],
            cluster_method=setting["cluster_method"],
            k=setting["k"],
            seed=setting["seed"],
            # FR
            te=setting["te"],
            wa=setting["wa"],
            we=setting["we"],
            fr_size_min=setting["fr_size_min"],
            fr_size_max=setting["fr_size_max"],
            fr_width=setting["fr_width"],
            edge_alpha=setting["edge_alpha"],
            # Aggregation
            key_node_lim=setting["key_node_lim"],
            key_node_method=setting["key_node_method"],
            agg_size_min=setting["agg_size_min"],
            agg_size_max=setting["agg_size_max"],
            agg_width_min=setting["agg_width_min"],
            agg_width_max=setting["agg_width_max"],
            agg_alpha=setting["agg_alpha"],
            fig_size=setting["fig_size"],
            ax_gap=setting["ax_gap"],
            random_click=setting["random_click"],
            itrac_mode=False,
        )
        break


run_process_tests()
