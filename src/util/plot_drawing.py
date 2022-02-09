import time
from src.util.graph_reading import (
    read_json_graph,
    read_facebook_graph,
    read_cora_graph,
    read_citeseer_graph,
    init_attributed_graph,
    get_degrees,
    clean_attributed_graph,
    read_miserables_graph,
    read_science_graph
)
from src.attr2vec.embedding import attributed_embedding
from src.layout.reduction import tsne
from src.layout.nx_fr import nx_fr
from src.layout.embedding_fr import embedding_fr
from src.util.cluster import kmeans, agglomerative
from src.util.normalize import normalize_list
import networkx as nx
import math
import matplotlib.pyplot as plt
import copy
from src.util.fish import mult_fish

COLOR_MAP = {
    0: "tab:blue",
    1: "tab:brown",
    2: "tab:orange",
    3: "mediumvioletred",
    4: "tab:green",
    5: "tab:red",
    6: "darkcyan",
    7: "tab:purple",
    8: "darkmagenta",
    9: "yellowgreen",
    10: "teal",
    11: "indianred",
    12: "salmon",
    13: "dodgerblue",
    14: "maroon",
}


def draw_attr2vec_tsne(
    graph_file,
    neighbor_weight=1,
    return_weight=1,
    attribute_weight=1,
    epochs=30,
    cluster_method="km",
    k=3,
    seed=1,
    size_min=8,
    size_max=16,
    width=0,
):
    if graph_file.startswith("facebook"):
        graph_file.strip("facebook")
        G = read_facebook_graph(graph_file)
    else:
        G = read_json_graph(graph_file)
    print("node num: ", len(list(G.nodes)))
    print("edge num: ", len(list(G.edges)))
    virtual_node_list = init_attributed_graph(G)
    print("node num in attri-graph: ", len(list(G.nodes)))
    print("edge num in attri-graph: ", len(list(G.edges)))
    degrees = get_degrees(G)
    sizes = [degree for degree in degrees]
    sizes = normalize_list(sizes, size_min, size_max)

    time1_start = time.time()
    vectors, weights = attributed_embedding(
        G,
        epochs=epochs,
        seed=seed,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        attribute_weight=attribute_weight,
        virtual_nodes=virtual_node_list,
        graph_name=graph_file)
    time1_end = time.time()
    print("Attr2Vec time: ", time1_end - time1_start)
    # pos = get_positions(G, vectors)
    time2_start = time.time()
    if cluster_method == "km":
        colors = kmeans(vectors, K=k)
    elif cluster_method == "agg":
        colors = agglomerative(vectors, K=k)
    elif cluster_method == "gt":
        colors = {}
        for node in G.nodes(data=True):
            if node[0].startswith("_"):
                continue
            colors[node[0]] = int(node[1]['group'])
    else:
        colors = kmeans(vectors, K=k)
    # colors = agglomerative(vectors, K=k)
    time2_end = time.time()
    print("Clustering time: ", time2_end - time2_start)
    print("Total time: ", time2_end - time1_start)
    clean_attributed_graph(G, vectors)
    pos = tsne(G, vectors)

    nodes = list(G.nodes)
    color_list = []
    # gt_color_list = []
    for node in nodes:
        if not node.startswith("_"):
            color_list.append(COLOR_MAP[colors[node]])
            # gt_color_list.append(node_groups_gt[node])
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=color_list)
    nx.draw_networkx_edges(G, pos, width=width)
    return vectors, weights, G, pos, colors


def draw_nx_fr(graph_file, color=None, size_min=8, size_max=8, width=0.5, edge_alpha=0.7):
    if graph_file.startswith("facebook"):
        graph_file.strip("facebook")
        G = read_facebook_graph(graph_file)
    else:
        G = read_json_graph(graph_file)
    degrees = get_degrees(G)
    sizes = [degree for degree in degrees]
    sizes = normalize_list(sizes, size_min, size_max)
    pos = nx_fr(G)
    if color:
        color_list = []
        nodes = list(G.nodes)
        for node in nodes:
            color_list.append(COLOR_MAP[color[node]])
    else:
        color_list = "b"
    nx.draw_networkx_nodes(G, pos, node_size=sizes,
                           node_color=color_list, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=width, alpha=edge_alpha)
    return G, pos


def draw_embedding_fr(
    graph_file,
    default_pos=None,
    return_weight=1,
    neighbor_weight=1,
    attribute_weight=1,
    epochs=30,
    k=3,
    seed=None,
    color=None,
    size_list=None,
    size_min=8, size_max=8,
    width=0.5, edge_alpha=0.7,
    te=0.6, wa=1, we=1,
    teh=0.85,
    fig=None,
    ax=None,
    ax_gap=None,
    fig_size=None,
    click_on=False,
    def_G=None
):
    # 读取图
    if graph_file.startswith("facebook"):
        graph_file.strip("facebook")
        G, class_map = read_facebook_graph()
    elif graph_file == "cora":
        G, class_map = read_cora_graph()
    elif graph_file == "miserables":
        G, class_map = read_miserables_graph()
    elif graph_file == "science":
        G, class_map = read_science_graph()
    elif graph_file == "citeseer":
        G, class_map = read_citeseer_graph()
    else:
        G = read_json_graph(graph_file)
    # Embedding
    virtual_node_list = init_attributed_graph(G)
    vectors, weights = attributed_embedding(
        G,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        attribute_weight=attribute_weight,
        epochs=epochs,
        seed=seed,
        virtual_nodes=virtual_node_list,
        graph_name=graph_file)
    clean_attributed_graph(G, vectors)
    # 计算大小
    if size_list:
        sizes = size_list
    else:
        degrees = get_degrees(G)
        sizes = [degree for degree in degrees]
        sizes = normalize_list(sizes, size_min, size_max)
    # 布局
    if default_pos:
        pos = default_pos
    else:
        pos = embedding_fr(G, vectors=vectors, te=te, wa=wa,
                           we=we, cluster=color, teh=teh)
    # 分配颜色
    if color:
        color_list = []
        nodes = list(G.nodes)
        for node in nodes:
            color_list.append(COLOR_MAP[color[node]])
    else:
        color = kmeans(G, vectors, K=k)
    # 计算标签
    labels = {}
    for node in nodes:
        labels[node] = node
    nx.draw_networkx_nodes(G, pos, node_size=sizes,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos, width=width, alpha=edge_alpha)
    if fig and ax and click_on:
        local_vectors, _ = attributed_embedding(
            G,
            neighbor_weight=10,
            epochs=epochs,
            seed=seed,
            virtual_nodes=[],
            graph_name=graph_file)
        global_vectors, _ = attributed_embedding(
            G,
            return_weight=10,
            epochs=epochs,
            seed=seed,
            virtual_nodes=[],
            graph_name=graph_file)
        virtual_node_list = init_attributed_graph(G)
        attr_vectors, _ = attributed_embedding(
            G,
            attribute_weight=10,
            epochs=epochs,
            seed=seed,
            virtual_nodes=virtual_node_list,
            graph_name=graph_file)
        clean_attributed_graph(G, attr_vectors)
        local_mat = __compute_similarity_matrix(local_vectors)
        global_mat = __compute_similarity_matrix(global_vectors)
        attr_mat = __compute_similarity_matrix(attr_vectors)
        offsets = {}
        current_proximity = [0]  # 0: local, 1: global, 2:attr
        for node in pos.keys():
            offsets[node] = (0, 0)
        fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: __on_node_click(
                event, ax, fig, ax_gap, fig_size, current_proximity,
                G, pos, sizes, color_list, width, edge_alpha,
                local_mat, global_mat, attr_mat, offsets
            ))
    return G, pos


def __on_node_click(event, ax, fig, ax_gap, fig_size, current_proximity,
                    G, pos, sizes, color_list, width, edge_alpha,
                    local_mat, global_mat, attr_mat, offsets):
    if not event.button == 1:
        return

    sel_node = __selected(event, pos, sizes, ax_gap,
                          fig_size, offsets, current_proximity)
    __redraw(ax, fig, ax_gap, fig_size, sel_node,
             G, pos, sizes, color_list, width, edge_alpha,
             local_mat, global_mat, attr_mat, offsets, current_proximity)


def __selected(event, pos, sizes, ax_gap, fig_size, offsets, current_proximity):
    click_xd = event.xdata
    click_yd = event.ydata

    if click_xd is None or click_yd is None:
        return None

    for i, node in enumerate(pos.keys()):
        node_pos_x = pos[node][0] + offsets[node][0]
        node_pos_y = pos[node][1] + offsets[node][1]
        dis = math.sqrt(((click_xd-node_pos_x))**2 +
                        ((click_yd-node_pos_y))**2)
        if isinstance(sizes, list):
            size = sizes[i]
        else:
            size = sizes
        r = __get_radius_from_size(size, ax_gap, fig_size)
        if dis <= r:
            return node
    return None


def __vector_distance(vec1, vec2):
    dis = 0
    for i in range(len(vec1)):
        dis += (vec1[i] - vec2[i])**2
    dis = math.sqrt(dis)
    return dis


def __compute_similarity_matrix(vectors):
    mat = {}
    min_dis = 99999
    max_dis = -1
    for node1 in vectors.keys():
        dis_dict = {}
        vec1 = vectors[node1]
        for node2 in vectors.keys():
            vec2 = vectors[node2]
            dis = __vector_distance(vec1, vec2)
            dis_dict[node2] = dis
            if dis > max_dis:
                max_dis = dis
            if dis < min_dis:
                min_dis = dis
        mat[node1] = dis_dict
    for node1 in mat.keys():
        for node2 in mat.keys():
            mat[node1][node2] = 1-(mat[node1][node2]-min_dis)/(max_dis-min_dis)
    return mat


def __search_similar_nodes(sel_node, local_mat, global_mat, sim_lim=0.85):
    local_list = []
    global_list = []
    items = local_mat[sel_node].items()
    sorted_items = sorted(items, key=lambda d: (d[1]), reverse=True)
    for node, sim in sorted_items:
        if node == sel_node:
            continue
        if sim < sim_lim:
            break
        local_list.append(node)
    items = global_mat[sel_node].items()
    sorted_items = sorted(items, key=lambda d: (d[1]), reverse=True)
    for node, sim in sorted_items:
        if node == sel_node:
            continue
        if sim < sim_lim:
            break
        global_list.append(node)
    return local_list, global_list


def __search_certain_similar_nodes(sel_node, mat, sim_lim=0.7, lim_number=3):
    sim_list = []
    items = mat[sel_node].items()
    sorted_items = sorted(items, key=lambda d: (d[1]), reverse=True)
    for node, sim in sorted_items:
        if len(sim_list) > lim_number:
            break
        if node == sel_node:
            continue
        if sim < sim_lim:
            break
        sim_list.append(node)
    return sim_list


def __search_neighbor_nodes(sel_node, G):
    neighbors = G.neighbors(sel_node)
    neighbor_list = []
    for neighbor in neighbors:
        neighbor_list.append(neighbor)
    return neighbor_list


def __process_similar_graphlet(local_sim, global_sim, sel_node, G, pos, sizes):
    sim_G = nx.Graph()
    sim_pos = {}
    sim_size = []
    sim_color = []
    nodes = list(G.nodes)
    for node in local_sim:
        sim_G.add_node(node)
        sim_pos[node] = pos[node]
        if isinstance(sizes, list):
            sim_size.append(sizes[nodes.index(node)])
        else:
            sim_size.append(sizes)
        if node in global_sim:
            sim_color.append("darkviolet")
        else:
            sim_color.append("red")
    for node in global_sim:
        if node in local_sim:
            continue
        sim_G.add_node(node)
        sim_pos[node] = pos[node]
        if isinstance(sizes, list):
            sim_size.append(sizes[nodes.index(node)])
        else:
            sim_size.append(sizes)
        sim_color.append("blue")
    sim_G.add_node(sel_node)
    sim_pos[sel_node] = pos[sel_node]
    if isinstance(sizes, list):
        sim_size.append(sizes[nodes.index(sel_node)])
    else:
        sim_size.append(sizes)
    sim_color.append("black")
    labels = {}
    for node in list(sim_G.nodes):
        labels[node] = node
    return sim_G, sim_pos, sim_size, sim_color, labels


def __process_certain_similar_graphlet(sim_list, sel_node, G, pos, sizes):
    sim_G = nx.Graph()
    sim_pos = {}
    sim_size = []
    sim_color = []
    nodes = list(G.nodes)
    for node in sim_list:
        sim_G.add_node(node)
        sim_pos[node] = pos[node]
        if isinstance(sizes, list):
            sim_size.append(sizes[nodes.index(node)])
        else:
            sim_size.append(sizes)
        sim_color.append("green")
    sim_G.add_node(sel_node)
    sim_pos[sel_node] = pos[sel_node]
    if isinstance(sizes, list):
        sim_size.append(sizes[nodes.index(sel_node)])
    else:
        sim_size.append(sizes)
    sim_color.append("red")
    labels = {}
    for node in list(sim_G.nodes):
        labels[node] = node
    return sim_G, sim_pos, sim_size, sim_color, labels


def __redraw(ax, fig, ax_gap, fig_size, sel_node,
             G, pos, sizes, color_list, width, edge_alpha,
             local_mat, global_mat, attr_mat, offsets, current_proximity):
    plt.cla()
    if sel_node is not None:
        if current_proximity[0] == 0:
            mat = local_mat
            print("Local proximity")
        elif current_proximity[0] == 1:
            mat = global_mat
            print("Global proximity")
        elif current_proximity[0] == 2:
            mat = attr_mat
            print("Attribute proximity")
        sim_list = __search_certain_similar_nodes(sel_node, mat)
        # sim_list = __search_neighbor_nodes(sel_node, G)
        sim_G, sim_pos, sim_size, sim_color, labels = __process_certain_similar_graphlet(
            sim_list, sel_node, G, pos, sizes
        )

        # local_sim, global_sim = __search_similar_nodes(
        #     sel_node, local_mat, global_mat)
        # sim_G, sim_pos, sim_size, sim_color, labels = __process_similar_graphlet(
        #     local_sim, global_sim, sel_node, G, pos, sizes)
        # cores = []
        # for node in local_sim:
        #     cores.append(node)
        # for node in global_sim:
        #     if node in local_sim:
        #         continue
        #     cores.append(node)

        cores = []
        for node in sim_list:
            cores.append(node)
        # __compute_fish_eye_offsets(cores, G, pos, offsets)
        for node in pos.keys():
            offsets[node] = (0, 0)
        final_pos = {}
        for node in pos.keys():
            final_pos[node] = (
                pos[node][0] + offsets[node][0],
                pos[node][1] + offsets[node][1]
            )
        # nx.draw_networkx_nodes(
        #     sim_G, final_pos, node_size=sim_size, node_color='none',
        #     edgecolors=sim_color, linewidths=1.5).set_zorder(3)
        current_color = []
        nodes = list(G.nodes)
        sim_nodes = list(sim_G.nodes)
        for i, node in enumerate(nodes):
            current_color.append("lightgray")
            if node not in sim_nodes:
                continue
            index = sim_nodes.index(node)
            current_color[i] = sim_color[index]
        label_pos = __label_offsets(final_pos, sizes, fig_size, ax_gap)
        nx.draw_networkx_labels(
            sim_G, label_pos, labels=labels, font_color='black', font_size=8)
        nx.draw_networkx_nodes(G, final_pos, node_size=sizes, node_color=current_color,
                               edgecolors='white', linewidths=0.7).set_zorder(2)
        nx.draw_networkx_edges(G, final_pos, width=width,
                               alpha=edge_alpha).set_zorder(1)
        current_proximity[0] = (current_proximity[0] + 1) % 3
    else:
        for node in offsets.keys():
            offsets[node] = (0, 0)
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=color_list,
                               edgecolors='white', linewidths=0.7).set_zorder(2)
        nx.draw_networkx_edges(G, pos, width=width,
                               alpha=edge_alpha).set_zorder(1)
        current_proximity[0] = 0

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig.canvas.draw()
    fig.canvas.flush_events()


def __label_offsets(pos, sizes, fig_size, ax_gap):
    label_pos = {}
    for i, node in enumerate(list(pos.keys())):
        size = sizes[i]
        r = __get_radius_from_size(size, ax_gap, fig_size)
        label_pos[node] = (
            pos[node][0],
            pos[node][1] - r - 0.008
        )
    return label_pos


def __normalize_node_postions(node_dict):
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for node in node_dict.values():
        x = node['x']
        y = node['y']
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    for node in node_dict.keys():
        x = node_dict[node]['x']
        y = node_dict[node]['y']
        node_dict[node] = {
            "x": (x - x_min) / (x_max - x_min) * 0.9 + 0.05,
            "y": (y - y_min) / (y_max - y_min) * 0.9 + 0.05
        }


def __get_radius_from_size(size, ax_gap, fig_size):
    return math.sqrt(size) / (2 * 72 * (1 - 2 * ax_gap) * fig_size)


def __get_size_from_radius(r, ax_gap, fig_size):
    k = (2 * 72 * (1 - 2 * ax_gap) * fig_size) ** 2
    return k * r ** 2


def __compute_fish_eye_offsets(cores, G, pos, offsets):
    G_dict = {'nodes': {}, 'links': []}
    for node in list(G.nodes):
        G_dict['nodes'][node] = {
            "x": pos[node][0],
            "y": pos[node][1]
        }
    for s, t in G.edges:
        G_dict['links'].append({
            'source': s,
            'target': t
        })
    nodes = list(pos.keys())
    temp_cores = []
    for core in cores:
        temp_cores.append(nodes.index(core))
    temp_nodes = {}
    temp_edges = []
    for node in G_dict['nodes'].keys():
        idx = nodes.index(node)
        temp_nodes[idx] = G_dict['nodes'][node]
    for link in G_dict['links']:
        idx_s = nodes.index(link['source'])
        idx_t = nodes.index(link['target'])
        temp_edges.append({
            'source': idx_s,
            'target': idx_t
        })
    temp = {
        'nodes': temp_nodes,
        'links': temp_edges
    }
    mult_fish(temp_cores, temp, ws=5, wr=1, wt=1)
    __normalize_node_postions(temp['nodes'])
    for node_idx in temp['nodes'].keys():
        node = nodes[node_idx]
        x, y = pos[node]
        offset_x = temp['nodes'][node_idx]['x'] - x
        offset_y = temp['nodes'][node_idx]['y'] - y
        offsets[node] = (offset_x, offset_y)
