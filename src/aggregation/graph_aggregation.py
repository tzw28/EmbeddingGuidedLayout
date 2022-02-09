import networkx as nx
import matplotlib.pyplot as plt
import math
from matplotlib.collections import LineCollection
from src.aggregation.curved_edges import curved_edges
import copy
from src.util.graph_reading import get_degrees
from src.util.normalize import normalize_list
from src.util.fish import mult_fish
import numpy as np
import warnings
import matplotlib.cbook
from src.layout.embedding_fr import embedding_fr

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def add_group_attr(G, groups, attribute='ag_group'):
    nodes = G.nodes
    for node in nodes.keys():
        group = groups[node]
        nodes[node][attribute] = group


class GraphAggregator:

    def __init__(self, G, pos,
                 group_attribute='group',
                 fig_size=10,
                 ax_gap=0.05,
                 is_curved=True,
                 attr_vectors=None,
                 local_vectors=None,
                 global_vectors=None,
                 weights=None,
                 key_node_method="degree",
                 key_node_lim=15):
        self.G = G
        self.nodes = G.nodes()
        self.edges = G.edges()
        pos_cp = copy.deepcopy(pos)
        self.group_attribute = group_attribute
        self.ag_G = None
        self.det_G = None
        self.selected_list = []
        self.selected_agg_size = {}
        self.selected_detail_node = None
        self.attribute_vectors = attr_vectors
        self.local_vectors = local_vectors
        self.global_vectors = global_vectors
        self.node_weights = weights
        self.fig_size = fig_size
        self.ax_gap = ax_gap
        self.key_node_method = key_node_method
        self.key_node_lim = key_node_lim
        self.agg_offsets = {}
        self.selected_agg_size_offsets = {}
        self.detail_offsets = {}
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        for p in pos_cp.values():
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]
            if p[1] < y_min:
                y_min = p[1]
            if p[1] > y_max:
                y_max = p[1]
        for node in pos_cp.keys():
            pos_cp[node][0] = (pos_cp[node][0] - x_min) / \
                (x_max - x_min) * 0.8 + 0.1
            pos_cp[node][1] = (pos_cp[node][1] - x_min) / \
                (x_max - x_min) * 0.8 + 0.1
        self.pos = pos_cp
        self.color_map = {
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
        self.is_curved = is_curved

    def __compute_aggregation_offsets(self):
        temp_G = nx.Graph()
        agg_edge_dict = self.aggregations['edges']
        for s, t in agg_edge_dict.keys():
            temp_G.add_edge(s, t)
        init_pos = {}
        for agg_node in self.aggregations['nodes'].keys():
            center = self.aggregations['nodes'][agg_node]['center']
            init_pos[agg_node] = (center[0], center[1])
        pos = nx.fruchterman_reingold_layout(
            temp_G, pos=init_pos, seed=14, iterations=1, k=1)
        agg_offsets = {}
        agg_x_min = 1
        agg_x_max = 0
        agg_y_min = 1
        agg_y_max = 0
        agg_x_min_node = None
        agg_x_max_node = None
        agg_y_min_node = None
        agg_y_max_node = None
        for node in pos.keys():
            center = pos[node]
            if center[0] < agg_x_min:
                agg_x_min = center[0]
                agg_x_min_node = node
            if center[0] > agg_x_max:
                agg_x_max = center[0]
                agg_x_max_node = node
            if center[1] < agg_y_min:
                agg_y_min = center[1]
                agg_y_min_node = node
            if center[1] > agg_y_max:
                agg_y_max = center[1]
                agg_y_max_node = node
        agg_x_min_lim = 0 + \
            self.__get_radius_from_size(self.agg_size[agg_x_min_node]) + 0.05
        agg_x_max_lim = 1 - \
            self.__get_radius_from_size(self.agg_size[agg_x_max_node]) - 0.05
        agg_y_min_lim = 0 + \
            self.__get_radius_from_size(self.agg_size[agg_y_min_node]) + 0.05
        agg_y_max_lim = 1 - \
            self.__get_radius_from_size(self.agg_size[agg_y_max_node]) - 0.05
        for node in pos.keys():
            center = pos[node]
            x1 = center[0]
            y1 = center[1]
            x2 = (x1 - agg_x_min) / (agg_x_max - agg_x_min) \
                * (agg_x_max_lim - agg_x_min_lim) + agg_x_min_lim
            y2 = (y1 - agg_y_min) / (agg_y_max - agg_y_min) \
                * (agg_y_max_lim - agg_y_min_lim) + agg_y_min_lim
            agg_offsets[node] = (x2 - init_pos[node][0],
                                 y2 - init_pos[node][1])
            # agg_offsets[node] = (0, 0)
        self.agg_offsets = agg_offsets
        return agg_offsets

    def __compute_detail_offsets(self, group):
        det_edges = self.det_G.edges
        temp_G = nx.Graph()
        init_pos = {}
        for s, t in det_edges:
            s_group = self.nodes[s][self.group_attribute]
            t_group = self.nodes[t][self.group_attribute]
            if s_group != group or t_group != group:
                continue
            temp_G.add_edge(s, t)
            init_pos[s] = (self.pos[s][0] + self.agg_offsets[group][0],
                           self.pos[s][1] + self.agg_offsets[group][1])
            init_pos[t] = (self.pos[t][0] + self.agg_offsets[group][0],
                           self.pos[t][1] + self.agg_offsets[group][1])
        for node in self.det_G.nodes:
            if node in init_pos.keys():
                continue
            node_group = self.nodes[node][self.group_attribute]
            if node_group != group:
                continue
            temp_G.add_node(node)
            init_pos[node] = (self.pos[node][0] + self.agg_offsets[group][0],
                              self.pos[node][1] + self.agg_offsets[group][1])

        # pos = nx.fruchterman_reingold_layout(temp_G, pos=init_pos, seed=14, iterations=5)
        pos = self.__graphlet_embedding_fr(temp_G, init_pos)
        # pos = init_pos
        pos = self.__normalize_positions(pos)
        # r = self.__get_radius_from_size(self.selected_agg_size[group])
        # 计算大小的偏置量
        b = 0.85
        r = self.__compute_selected_aggs_radius(group)
        new_s = self.__get_size_from_radius(r)
        # print(self.selected_agg_size)
        self.selected_agg_size_offsets[group] = new_s - \
            self.selected_agg_size[group]
        for node in pos.keys():
            x, y = pos[node]
            center = self.aggregations['nodes'][group]['center']
            new_x = (b*r) * x * math.sqrt(1-0.5*y**2) + \
                center[0] + self.agg_offsets[group][0]
            new_y = (b*r) * y * math.sqrt(1-0.5*x**2) + \
                center[1] + self.agg_offsets[group][1]
            pos[node] = np.array([new_x, new_y])
        for node in init_pos.keys():
            self.detail_offsets[node] = (pos[node][0] - init_pos[node][0],
                                         pos[node][1] - init_pos[node][1])
        return

    def __compute_detail_offsets_with_viturals(self, group, det_G, det_pos, vir_G, vir_pos, det_color, out_edges):
        temp_G = nx.Graph()
        temp_pos = {}
        fixed = []
        for (s, t) in det_G.edges:
            group_s = self.nodes[s][self.group_attribute]
            if group_s != group:
                continue
            temp_G.add_edge(s, t)
            temp_pos[s] = det_pos[s]
            temp_pos[t] = det_pos[t]
        for (s, t) in vir_G.edges:
            if len(s.split("_")) != 3:
                group_st = self.nodes[s][self.group_attribute]
            else:
                group_st = self.nodes[t][self.group_attribute]
            if group_st != group:
                continue
            if len(s.split("_")) == 3:
                fixed.append(s)
            else:
                fixed.append(t)
            temp_G.add_edge(s, t)
            temp_pos[s] = vir_pos[s]
            temp_pos[t] = vir_pos[t]
        # print(len(list(temp_G.nodes)))
        colors = {}
        for node in det_G.nodes:
            g = self.nodes[node][self.group_attribute]
            if g != group:
                continue
            colors[node] = g
        for s, t in vir_G.edges:
            strs_s = s.split("_")
            strs_t = t.split("_")
            if len(strs_s) == 3:
                g = self.nodes[t][self.group_attribute]
                if g != group:
                    continue
                nbr_g = int(strs_s[0]) if strs_s[2] == '1' else int(strs_s[1])
                colors[t] = nbr_g
                colors[s] = nbr_g
            else:
                g = self.nodes[s][self.group_attribute]
                if g != group:
                    continue
                nbr_g = int(strs_t[0]) if strs_t[2] == '1' else int(strs_t[1])
                colors[s] = nbr_g
                colors[t] = nbr_g
        # new_temp_pos = nx.fruchterman_reingold_layout(temp_G, pos=temp_pos, fixed=fixed, seed=17, scale=0.1)
        for (s, t) in out_edges:
            group_s = self.nodes[s][self.group_attribute]
            group_t = self.nodes[t][self.group_attribute]
            if group_s != group and group_t != group:
                continue
            if group_s == group:
                vir_node = "{}_{}_0".format(group_s, group_t)
                self.attribute_vectors[vir_node] = copy.deepcopy(
                    self.attribute_vectors[t])
                vir_node = "{}_{}_1".format(group_t, group_s)
                self.attribute_vectors[vir_node] = copy.deepcopy(
                    self.attribute_vectors[t])
            if group_t == group:
                vir_node = "{}_{}_0".format(group_t, group_s)
                self.attribute_vectors[vir_node] = copy.deepcopy(
                    self.attribute_vectors[s])
                vir_node = "{}_{}_1".format(group_s, group_t)
                self.attribute_vectors[vir_node] = copy.deepcopy(
                    self.attribute_vectors[s])
        temp_pos = self.__normalize_positions(temp_pos)
        new_temp_pos = self.__graphlet_embedding_fr(
            temp_G, pos=temp_pos, fixed=fixed,
            cluster=colors, tel=0.6, teh=0.99
        )
        new_temp_pos = self.__normalize_positions(new_temp_pos)
        sel_group_size = self.selected_agg_size[group] + \
            self.selected_agg_size_offsets[group]
        r = self.__get_radius_from_size(sel_group_size)
        b = 0.9
        center = self.aggregations['nodes'][group]['center']
        for node in list(new_temp_pos.keys()):
            if node in fixed:
                new_temp_pos.pop(node)
                continue
            x, y = new_temp_pos[node]
            new_x = (b*r) * x * math.sqrt(1-0.5*y**2) + \
                center[0] + self.agg_offsets[group][0]
            new_y = (b*r) * y * math.sqrt(1-0.5*x**2) + \
                center[1] + self.agg_offsets[group][1]
            new_temp_pos[node] = np.array([new_x, new_y])
        for node in new_temp_pos.keys():
            self.detail_offsets[node] = (new_temp_pos[node][0] - det_pos[node][0],
                                         new_temp_pos[node][1] - det_pos[node][1])

    def __compute_offsets(self):
        self.__compute_aggregation_offsets()

    def __compute_fish_eye_offsets(self):
        return
        agg_G = {
            'nodes': {},
            'links': []
        }
        agg_node_dict = self.aggregations['nodes']
        for node in agg_node_dict.keys():
            x = agg_node_dict[node]['center'][0] + self.agg_offsets[node][0]
            y = agg_node_dict[node]['center'][1] + self.agg_offsets[node][1]
            agg_G['nodes'][node] = {
                "x": x,
                "y": y
            }
        agg_edge_dict = self.aggregations['edges']
        for edge in agg_edge_dict.keys():
            agg_G['links'].append({
                "source": edge[0],
                "target": edge[1]
            })
        temp_G = copy.deepcopy(agg_G)
        mult_fish(self.selected_list, temp_G, ws=5, wr=1, wt=1)
        for node in temp_G['nodes']:
            x, y = self.agg_offsets[node]
            new_x = temp_G['nodes'][node]['x'] - agg_G['nodes'][node]['x']
            new_y = temp_G['nodes'][node]['y'] - agg_G['nodes'][node]['y']
            # self.agg_offsets[node] = (new_x, new_y)
        return

    def __compute_selected_aggs_radius(self, agg_node):
        neighbors = []
        for s, t in self.aggregations['edges'].keys():
            if s != agg_node and t != agg_node:
                continue
            if s == agg_node:
                neighbors.append(t)
            else:
                neighbors.append(s)
        agg_node_dict = self.aggregations['nodes']
        current_center = (
            agg_node_dict[agg_node]['center'][0] +
            self.agg_offsets[agg_node][0],
            agg_node_dict[agg_node]['center'][1] +
            self.agg_offsets[agg_node][1]
        )
        gap = 0.03
        lim = (0.1, 0.9)
        current_count = agg_node_dict[agg_node]['count']
        min_r = 1
        for n in neighbors:
            count = agg_node_dict[n]['count']
            rate = current_count / (current_count + count)
            if rate > lim[1]:
                rate = lim[1]
            if rate < lim[0]:
                rate = lim[0]
            rate -= gap
            center = (
                agg_node_dict[n]['center'][0] + self.agg_offsets[n][0],
                agg_node_dict[n]['center'][1] + self.agg_offsets[n][1]
            )
            dis = self.__distance(current_center, center, method="eu")
            # r = dis * rate
            r = dis * 0.65
            if r < min_r:
                min_r = r
        if current_center[0] - min_r < 0:
            min_r = current_center[0] - 0.01
        if current_center[0] + min_r > 1:
            min_r = 1 - current_center[0] - 0.01
        if current_center[1] - min_r < 0:
            min_r = current_center[1] - 0.01
        if current_center[1] + min_r > 1:
            min_r = 1 - current_center[1] - 0.01
        return min_r

    def __graphlet_embedding_fr(self, G, pos=None, cluster=None, fixed=None,
                                tel=0.6, teh=0.85):
        temp_vectors = {}
        for node in list(G.nodes):
            temp_vectors[node] = self.attribute_vectors[node]
        pos = embedding_fr(G, temp_vectors, pos=pos, cluster=cluster,
                           tel=tel, teh=teh)
        return pos

    def __distance(self, vec1, vec2, method):
        if method == "eu":
            dis = 0
            for i in range(len(vec1)):
                dis += (vec1[i] - vec2[i])**2
            dis = math.sqrt(dis)
        return dis

    def __get_similar_nodes(self, node, sim_type="local", dis_method="eu", node_num=3):
        if sim_type == "local":
            vectors = self.local_vectors
        elif sim_type == "global":
            vectors = self.global_vectors
        else:
            vectors = self.attribute_vectors
        vec1 = vectors[node]
        dis_list = []
        for det_node in self.det_G.nodes:
            if det_node == node or det_node.startswith("vkn_"):
                continue
            vec2 = vectors[det_node]
            dis = self.__distance(vec1, vec2, dis_method)
            dis_list.append((det_node, dis))
        sorted_dis_list = sorted(dis_list, key=lambda d: d[1], reverse=False)
        similar_nodes = sorted_dis_list[:node_num]
        similar_nodes = [node[0] for node in similar_nodes]
        return similar_nodes

    def __normalize_list(self, l):
        sum = 0.0
        for x in l:
            sum += x
        nl = []
        for x in l:
            nl.append(float(x) / sum)
        return nl

    def __normalize_positions(self, pos, x_range=(-1, 1), y_range=(-1, 1)):
        nm_pos = {}
        x_min = 100
        x_max = -100
        y_min = 100
        y_max = -100
        for node in pos.keys():
            p = pos[node]
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]
            if p[1] < y_min:
                y_min = p[1]
            if p[1] > y_max:
                y_max = p[1]
        for node in pos.keys():
            x1 = pos[node][0]
            y1 = pos[node][1]
            x2 = (x1 - x_min) / (x_max - x_min) * \
                 (x_range[1] - x_range[0]) + x_range[0]
            y2 = (y1 - y_min) / (y_max - y_min) * \
                 (y_range[1] - y_range[0]) + y_range[0]
            nm_pos[node] = (x2, y2)
        return nm_pos

    def __get_radius_from_size(self, size):
        return math.sqrt(size) / (2 * 72 * (1 - 2 * self.ax_gap) * self.fig_size)

    def __get_size_from_radius(self, r):
        k = (2 * 72 * (1 - 2 * self.ax_gap) * self.fig_size) ** 2
        return k * r ** 2

    def __get_relative_with(self, width):
        return width / (72 * (1 - 2 * self.ax_gap) * self.fig_size)

    def __get_absolute_with(self, width):
        return width * (72 * (1 - 2 * self.ax_gap) * self.fig_size)

    # 从边列表中查找与指定节点相连的边
    def __get_edges(self, node, edges):
        result = []
        for (s, t) in edges:
            if s != node and t != node:
                continue
            result.append((s, t))
        return result

    def __rearrange_linear(self, x, x0, x1, y0, y1):
        y = (x-x0) / (x1-x0) * (y1-y0) + y0
        return y

    def __rearrange_edge_width(self, width_list):
        w_min = width_list[0]
        w_max = width_list[0]
        for w in width_list:
            if w < w_min:
                w_min = w
            if w > w_max:
                w_max = w
        ag_nodes = self.aggregations['nodes']
        for i, (s, t) in enumerate(self.ag_G.edges):
            w = width_list[i]
            size_s = self.s1 + ag_nodes[s]['count'] * self.s2
            size_t = self.s1 + ag_nodes[t]['count'] * self.s2
            if size_s < size_t:
                size = size_s
            else:
                size = size_t
            r = self.__get_radius_from_size(size)
            r = self.__get_absolute_with(r)
            width_list[i] = self.__rearrange_linear(
                w, w_min, w_max, w_min, 2*r)

    def __get_sorted_nodes_degree(self, G):
        detail_node_list = []
        for i, node in enumerate(G.nodes):
            detail_node = {
                'id': node,
                'group': self.nodes[node][self.group_attribute],
                'degree': self.G.degree[node]
            }
            detail_node_list.append(detail_node)
        sorted_node_list = sorted(detail_node_list, key=lambda d: (
            d['group'], d['degree']), reverse=True)
        return sorted_node_list

    def __get_sorted_nodes_tfidf(self, G):
        detail_node_list = []
        for i, node in enumerate(G.nodes):
            detail_node = {
                'id': node,
                'group': self.nodes[node][self.group_attribute],
                'weight': self.node_weights[node]
            }
            detail_node_list.append(detail_node)
        sorted_node_list = sorted(detail_node_list, key=lambda d: (
            d['group'], d['weight']), reverse=True)
        return sorted_node_list

    def __key_nodes(self, G, method="degree"):
        if method == "tfidf" and self.node_weights is not None:
            sorted_node_list = self.__get_sorted_nodes_tfidf(G)
        else:
            sorted_node_list = self.__get_sorted_nodes_degree(G)
        group_nodes = {}
        # 遍历所有细节图的所有节点
        for node in sorted_node_list:
            if node['group'] not in group_nodes.keys():
                group_nodes[node['group']] = []
            is_out_edge_node = False
            for (s, t) in self.edges:
                if s != node['id'] and t != node['id']:
                    continue
                group_s = self.nodes[s][self.group_attribute]
                group_t = self.nodes[t][self.group_attribute]
                if group_s == group_t:
                    continue
                is_out_edge_node = True
                break
            if is_out_edge_node:
                continue
            group_nodes[node['group']].append(node['id'])
        for group in group_nodes.keys():
            group_nodes[group] = group_nodes[group][:self.key_node_lim]
        return group_nodes

    def __selected(self, event):
        click_xd = event.xdata
        click_yd = event.ydata

        if click_xd is None or click_yd is None:
            return None, None

        # 点击聚合节点
        ag_nodes = self.aggregations['nodes']
        for node in ag_nodes.keys():
            node_info = ag_nodes[node]
            nxd = node_info['center'][0] + self.agg_offsets[node][0]
            nyd = node_info['center'][1] + self.agg_offsets[node][1]
            # dis = math.sqrt(((click_xd-nxd)*width)**2 + ((click_yd - nyd)*height)**2) * dpi
            dis = math.sqrt(((click_xd-nxd))**2 + ((click_yd - nyd))**2)
            size = self.agg_size[node]
            r = self.__get_radius_from_size(size)
            if dis > r:
                continue
            if not self.ag_G.has_node(node):
                break
            return node, "agg"
        # 点击细节节点
        if not self.det_G:
            return None, None
        for i, det_node in enumerate(self.det_G.nodes):
            node_pos_x = self.det_pos[det_node][0]
            node_pos_y = self.det_pos[det_node][1]
            dis = math.sqrt(((click_xd-node_pos_x))**2 +
                            ((click_yd-node_pos_y))**2)
            size = self.det_size[i]
            r = self.__get_radius_from_size(size)
            if dis <= r:
                return det_node, "det"
        self.selected_detail_node = None
        return None, None

    def __on_node_click(self, event):
        if not event.button == 1:
            return

        sel_node_id, node_type = self.__selected(event)
        fig = event.canvas.figure
        if sel_node_id is None:
            self.__redraw(fig)
            return
        if node_type == "agg":
            print("remove {}".format(sel_node_id))
            self.ag_G.remove_node(sel_node_id)
            self.selected_list.append(sel_node_id)
            self.__compute_fish_eye_offsets()
        if node_type == "det":
            print("select {}".format(sel_node_id))
            self.selected_detail_node = sel_node_id
        self.__redraw(fig)

    def __process_basic_graphlet(self):
        ag_nodes = self.aggregations['nodes']
        ag_edges = self.aggregations['edges']
        nodes = self.ag_G.nodes
        edges = self.ag_G.edges
        pos = {}
        size = {}
        color = {}
        label = {}
        if self.node_weights is not None:
            node_weight_list = [(key, self.node_weights[key])
                                for key in self.node_weights.keys()]
            sorted_node_list = sorted(
                node_weight_list, key=lambda d: d[1], reverse=True)
            for (node, w) in sorted_node_list:
                if node.startswith("_") or node == "nan":
                    continue
                group = self.nodes[node][self.group_attribute]
                if group in label.keys() or group not in nodes.keys():
                    continue
                label[group] = node
        for node in nodes.keys():
            pos[node] = (ag_nodes[node]['center'][0] + self.agg_offsets[node][0],
                         ag_nodes[node]['center'][1] + self.agg_offsets[node][1])
            size[node] = self.agg_size[node]
            color[node] = self.color_map[node]
            if label is {}:
                label[node] = node
        width = []
        for (s, t) in edges:
            width.append(self.agg_width[(s, t)])
        color_list = list(color.values())
        size_list = list(size.values())
        return pos, color_list, size_list, width, label

    def __process_selected_graphlet(self, det_nodes, det_pos):
        '''
        计算所选聚合节点的子图
        '''
        ag_nodes = self.aggregations['nodes']
        ag_edges = self.aggregations['edges']
        sel_G = nx.Graph()
        pos = {}
        size = {}
        color = {}
        width = []
        for (s, t) in ag_edges.keys():
            if s not in self.selected_list and t not in self.selected_list:
                continue
            sel_G.add_edge(s, t)
        '''
        for (s, t) in sel_G.edges:
            try:
                count = ag_edges[(s, t)]['count']
            except KeyError:
                count = ag_edges[(t, s)]['count']
            width.append(self.w1 + count * self.w2)
        '''
        for (s, t) in sel_G.edges:
            try:
                width.append(self.agg_width[(s, t)])
            except KeyError:
                width.append(self.agg_width[(t, s)])

        for node in list(sel_G.nodes):
            pos[node] = (ag_nodes[node]['center'][0] + self.agg_offsets[node][0],
                         ag_nodes[node]['center'][1] + self.agg_offsets[node][1])
            color[node] = self.color_map[node]
            # color[node] = node
            if node in self.selected_list:
                size[node] = self.selected_agg_size[node] + \
                    self.selected_agg_size_offsets[node]
            else:
                size[node] = self.agg_size[node]
            '''
            size[node] = self.agg_size[node]
            if node in self.selected_list:
                r = self.__get_radius_from_size(size[node])
                for det_node in det_nodes:
                    if det_node.startswith("vkn_"):
                        group = int(det_node.split("_")[1])
                    else:
                        group = self.nodes[det_node][self.group_attribute]
                    if group != node:
                        continue
                    dp = det_pos[det_node]
                    dis = math.sqrt((dp[0]-pos[node][0])**2 + (dp[1]-pos[node][1])**2)
                    if dis > r:
                        r = dis + 0.01
                        s = self.__get_size_from_radius(r)
                        size[node] = s
                        self.selected_agg_size[node] = s
            '''
        color_list = list(color.values())
        size_list = list(size.values())
        return sel_G, pos, color_list, size_list, width, size

    def __process_relative_nodes(self):
        rel_G = nx.Graph()
        sel_det_node = self.selected_detail_node
        if sel_det_node is None:
            return False, rel_G, {}, [], []
        pos = {}
        size = {}
        color = {}
        det_nodes = list(self.det_G.nodes)
        local_similar_nodes = self.__get_similar_nodes(
            sel_det_node, sim_type="local")
        global_similar_nodes = self.__get_similar_nodes(
            sel_det_node, sim_type="global")
        for node in local_similar_nodes:
            rel_G.add_node(node)
            pos[node] = self.det_pos[node]
            size[node] = self.det_size[det_nodes.index(node)]
            color[node] = "red"
        for node in global_similar_nodes:
            rel_G.add_node(node)
            pos[node] = self.det_pos[node]
            size[node] = self.det_size[det_nodes.index(node)]
            if node in local_similar_nodes:
                color[node] = "darkviolet"
            else:
                color[node] = "blue"
        similar_nodes = self.__get_similar_nodes(sel_det_node)
        for node in similar_nodes:
            rel_G.add_node(node)
            pos[node] = self.det_pos[node]
            size[node] = self.det_size[det_nodes.index(node)]
        rel_G.add_node(sel_det_node)
        pos[sel_det_node] = self.det_pos[sel_det_node]
        size[sel_det_node] = self.det_size[det_nodes.index(sel_det_node)]
        color[sel_det_node] = "slategray"
        size_list = []
        color_list = []
        for node in rel_G.nodes:
            size_list.append(size[node])
            color_list.append(color[node])
        return True, rel_G, pos, size_list, color_list

    def __process_detail_graphlet(self, select_key_nodes=False):
        det_G = nx.Graph()
        pos = {}
        color = {}
        nodes = []
        edges = []
        out_edges = []
        # 遍历原图全部节点一遍，找到选中的组内的节点
        for node in self.nodes.keys():
            group = self.nodes[node][self.group_attribute]
            if group not in self.selected_list:
                continue
            nodes.append(node)
            pos[node] = (self.pos[node][0] + self.agg_offsets[group][0],
                         self.pos[node][1] + self.agg_offsets[group][1])
            color[node] = self.color_map[group]
            # color[node] = group
            det_G.add_node(node)
        current_group = self.selected_list[-1]
        ag_nodes = self.aggregations['nodes']
        if current_group not in self.selected_agg_size.keys():
            center = (ag_nodes[current_group]['center'][0] + self.agg_offsets[current_group][0],
                      ag_nodes[current_group]['center'][1] + self.agg_offsets[current_group][1])
            # s = self.agg_size[current_group]
            # r = self.__get_radius_from_size(s)
            r = 0
            for node in list(det_G.nodes):
                if node.startswith("vkn_"):
                    group = int(node.split("_")[1])
                else:
                    group = self.nodes[node][self.group_attribute]
                if group != current_group:
                    continue
                dp = pos[node]
                dis = math.sqrt((dp[0]-center[0])**2 + (dp[1]-center[1])**2)
                if dis > r:
                    r = dis + 0.01
                    s = self.__get_size_from_radius(r)
                    self.selected_agg_size[current_group] = s
        # 遍历原图全部边一遍，找到组之间的连边、组内边
        for (s, t) in self.edges:
            group_s = self.nodes[s][self.group_attribute]
            group_t = self.nodes[t][self.group_attribute]
            if group_s != group_t:
                out_edges.append((s, t))
                # if s in nodes and group_s in self.selected_list:
                #     color[s] = self.color_map[group_t]
                # elif t in nodes and group_t in self.selected_list:
                #     color[t] = self.color_map[group_s]
                continue
            if s not in nodes or t not in nodes:
                continue
            edges.append((s, t))
            det_G.add_edge(s, t)
        # 根据度数计算内部节点的大小
        nodes = det_G.nodes
        degrees = []
        for node in nodes:
            degrees.append(det_G.degree[node])
        if len(list(self.G.nodes)) > 1000:
            sizes = [20 for degree in degrees]
        elif len(list(self.G.nodes)) > 300:
            sizes = [50 + 2*degree for degree in degrees]
        else:
            sizes = [40 + 15*degree for degree in degrees]
        out_edges = list(set(out_edges))
        # 去重
        for (s, t) in out_edges:
            if (t, s) in out_edges:
                out_edges.remove((t, s))
        # color_list = list(color.values())
        color_list = [color[node] for node in det_G.nodes]
        if select_key_nodes:
            # 挑选关键节点
            key_nodes = self.__key_nodes(det_G, method=self.key_node_method)

            # 根据关键节点重新布局
            node_list = list(det_G.nodes)
            for node in node_list:
                group = self.nodes[node][self.group_attribute]
                if len(key_nodes[group]) == 0:
                    continue
                if node in key_nodes[group]:
                    continue
                # 不关键节点的集合节点，vkn_group
                vir_key_node = "vkn_{}".format(group)
                det_G.add_node(vir_key_node)
                pos[vir_key_node] = np.array(
                    [self.aggregations['nodes'][group]['center'][0] + self.agg_offsets[group][0],
                     self.aggregations['nodes'][group]['center'][1] + self.agg_offsets[group][1]])
                color[vir_key_node] = self.color_map[group]
                # 遍历内部边，内部边连接到集合节点上
                edge_list = self.__get_edges(node, det_G.edges)
                for (s, t) in edge_list:
                    if s == node:
                        det_G.add_edge(vir_key_node, t)
                    elif t == node:
                        det_G.add_edge(s, vir_key_node)
                # 遍历外部边，如果有外部边则保留这个节点
                out_tag = False
                for (s, t) in out_edges:
                    if s == node or t == node:
                        out_tag = True
                # 移除不关键节点
                if not out_tag:
                    det_G.remove_node(node)
                    color.pop(node)
                    pos.pop(node)
            nodes = det_G.nodes
            degrees = []
            for node in nodes:
                degrees.append(det_G.degree[node])
            sizes = [16 + 2 * degree for degree in degrees]
            for i, node in enumerate(nodes):
                if node.startswith("vkn_"):
                    sizes[i] = 3
            color_list = list(color.values())
        self.det_G = det_G
        self.det_pos = pos
        self.det_size = sizes
        self.__compute_detail_offsets(group=self.selected_list[-1])
        for node in pos.keys():
            pos[node] = (pos[node][0] + self.detail_offsets[node][0],
                         pos[node][1] + self.detail_offsets[node][1])
        return det_G, pos, color_list, sizes, out_edges

    def __process_virtual_graphlet(self, fig, det_pos, sel_size, out_edges, det_G, det_color):
        vir_G = nx.Graph()
        pos = {}
        vir_pos = {}
        vir_color = {}
        for (s, t) in out_edges:
            group_s = self.nodes[s][self.group_attribute]
            group_t = self.nodes[t][self.group_attribute]
            if group_s not in self.selected_list and group_t not in self.selected_list:
                continue
            # 计算虚拟节点位置
            if group_s in self.selected_list:
                det_node = s
                vir_node = "{}_{}_0".format(group_s, group_t)
                if not vir_G.has_node(vir_node):
                    sel_center = (self.aggregations['nodes'][group_s]['center'][0] + self.agg_offsets[group_s][0],
                                  self.aggregations['nodes'][group_s]['center'][1] + self.agg_offsets[group_s][1])
                    nbr_center = (self.aggregations['nodes'][group_t]['center'][0] + self.agg_offsets[group_t][0],
                                  self.aggregations['nodes'][group_t]['center'][1] + self.agg_offsets[group_t][1])
                    r = self.__get_radius_from_size(sel_size[group_s])
                    l = math.sqrt((nbr_center[0] - sel_center[0]) ** 2 +
                                  (nbr_center[1] - sel_center[1]) ** 2)
                    x = (nbr_center[0] - sel_center[0]) * r / l + sel_center[0]
                    y = (nbr_center[1] - sel_center[1]) * r / l + sel_center[1]
                    pos[(group_s, group_t)] = (x, y)
                    vir_pos[vir_node] = (x, y)
                    vir_G.add_node(vir_node)
                # 计算虚拟边
                # vir_pos[det_node] = (det_pos[det_node][0] + self.agg_offsets[group_s][0],
                #                     det_pos[det_node][1] + self.agg_offsets[group_s][1])
                vir_pos[det_node] = (det_pos[det_node][0],
                                     det_pos[det_node][1])
                vir_G.add_node(det_node)
                vir_G.add_edge(det_node, vir_node)
                vir_color[(det_node, vir_node)] = group_t
            if group_t in self.selected_list:
                det_node = t
                vir_node = "{}_{}_1".format(group_s, group_t)
                if not vir_G.has_node(vir_node):
                    sel_center = (self.aggregations['nodes'][group_t]['center'][0] + self.agg_offsets[group_t][0],
                                  self.aggregations['nodes'][group_t]['center'][1] + self.agg_offsets[group_t][1])
                    nbr_center = (self.aggregations['nodes'][group_s]['center'][0] + self.agg_offsets[group_s][0],
                                  self.aggregations['nodes'][group_s]['center'][1] + self.agg_offsets[group_s][1])
                    r = self.__get_radius_from_size(sel_size[group_t])
                    l = math.sqrt((nbr_center[0] - sel_center[0]) ** 2 +
                                  (nbr_center[1] - sel_center[1]) ** 2)
                    x = (nbr_center[0] - sel_center[0]) * r / l + sel_center[0]
                    y = (nbr_center[1] - sel_center[1]) * r / l + sel_center[1]
                    pos[(group_s, group_t)] = (x, y)
                    vir_pos[vir_node] = (x, y)
                    vir_G.add_node(vir_node)
                # 计算虚拟边
                # vir_pos[det_node] = (det_pos[det_node][0] + self.agg_offsets[group_t][0],
                #                      det_pos[det_node][1] + self.agg_offsets[group_t][1])
                vir_pos[det_node] = (det_pos[det_node][0],
                                     det_pos[det_node][1])
                vir_G.add_node(det_node)
                vir_G.add_edge(det_node, vir_node)
                vir_color[(det_node, vir_node)] = group_s
        color_list = []
        for (s, t) in vir_G.edges:
            strs = s.split("_")
            if len(strs) != 3:
                strs = t.split("_")
            # print(strs)
            if int(strs[2]) == 0:
                color_list.append(self.color_map[int(strs[1])])
            elif int(strs[2]) == 1:
                color_list.append(self.color_map[int(strs[0])])
        current_group = self.selected_list[-1]
        offsets_copy = copy.deepcopy(self.detail_offsets)
        for node in det_pos.keys():
            det_pos[node] = (det_pos[node][0] - offsets_copy[node][0],
                             det_pos[node][1] - offsets_copy[node][1])
        self.__compute_detail_offsets_with_viturals(
            current_group, det_G, det_pos, vir_G, vir_pos, det_color, out_edges)
        new_det_pos = {}
        for node in det_G.nodes:
            new_det_pos[node] = (det_pos[node][0] + self.detail_offsets[node][0],
                                 det_pos[node][1] + self.detail_offsets[node][1])
            vir_pos[node] = new_det_pos[node]
        return vir_G, vir_pos, color_list, new_det_pos

    def __redraw(self, fig):
        # 计算
        bsc_pos, bsc_color, bsc_size, bsc_width, bsc_label = self.__process_basic_graphlet()
        det_G, det_pos, det_color, det_size, out_edges = self.__process_detail_graphlet()
        sel_G, sel_pos, sel_color, sel_size, sel_width, sel_size_dict = self.__process_selected_graphlet(
            det_G.nodes, det_pos)
        vir_G, vir_pos, vir_color, det_pos = self.__process_virtual_graphlet(
            fig, det_pos, sel_size_dict, out_edges, det_G, det_color)
        is_selected, rel_G, rel_pos, rel_size, rel_color = self.__process_relative_nodes()

        # 开始重绘
        plt.cla()
        gap = self.ax_gap
        ax = plt.axes([gap, gap, 1 - gap * 2, 1 - gap * 2])
        ax.set_aspect(1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # 绘制所选节点白色图
        nx.draw_networkx_nodes(
            sel_G, sel_pos, node_size=sel_size, node_color='white', alpha=1)
        nx.draw_networkx_edges(
            sel_G, sel_pos, width=sel_width, edge_color='gray', alpha=0.4)

        # 绘制所选节点透明图
        # nx.draw_networkx_nodes(sel_G, sel_pos, node_size=sel_size, node_color='none', linewidths=2, edgecolors=sel_color, alpha=0.5)
        nx.draw_networkx_nodes(sel_G, sel_pos, node_size=sel_size, node_color=sel_color, style="dashed",
                               linewidths=2, edgecolors=sel_color, alpha=self.agg_alpha, )
        nx.draw_networkx_edges(
            sel_G, sel_pos, width=sel_width, edge_color='gray', alpha=0.4)

        # 绘制其他点和边
        nx.draw_networkx_nodes(
            self.ag_G, bsc_pos, node_size=bsc_size, node_color=bsc_color)
        # nx.draw_networkx_labels(self.ag_G, bsc_pos, labels=bsc_label, font_size=10)
        nx.draw_networkx_edges(
            self.ag_G, bsc_pos, width=bsc_width, edge_color='gray', alpha=0.4)

        # 绘制选中部分
        nx.draw_networkx_nodes(det_G, det_pos, node_size=det_size,
                               node_color=det_color, edgecolors='white', linewidths=0.8).set_zorder(6)
        '''
        labels = {}
        for node in det_G.nodes:
            labels[str(node)] = str(node)
        nx.draw_networkx_labels(det_G, det_pos, labels=labels)
        '''
        if len(list(det_G.edges)) > 0:
            nx.draw_networkx_edges(
                det_G, det_pos, width=0.5, edge_color='black', alpha=0.2).set_zorder(2)

        # 绘制虚拟边
        nx.draw_networkx_nodes(vir_G, vir_pos, node_size=50,
                               node_color='black', alpha=0).set_zorder(4)
        if self.is_curved:
            curves = curved_edges(vir_G, vir_pos, polarity='graphlet',
                                  dist_ratio=0.13, centers=self.aggregations['nodes'])
            lc = LineCollection(curves, linewidths=1,
                                color=vir_color, alpha=0.5)  # .set_zorder(5)
            ax.add_collection(lc)
        else:
            nx.draw_networkx_edges(
                vir_G, vir_pos, width=1, edge_color='gray', alpha=0.7).set_zorder(5)

        # 绘制选中细节节点和相关节点
        if is_selected:
            '''
            labels = {}
            for node in rel_G.nodes:
                labels[str(node)] = str(node)
            nx.draw_networkx_labels(rel_G, rel_pos, labels=labels)
            '''
            nx.draw_networkx_nodes(rel_G, rel_pos, node_size=rel_size, linewidths=1.5,
                                   edgecolors=rel_color, alpha=1, node_color='none').set_zorder(7)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def generate_aggregations(self):
        groups = {}
        ga = self.group_attribute
        for node in self.nodes.keys():
            g = self.nodes[node][ga]
            if g not in groups.keys():
                groups[g] = [node]
            else:
                groups[g].append(node)
        aggregate_nodes = {}
        for group in groups.keys():
            center_x = 0.0
            center_y = 0.0
            count = 0
            for node in groups[group]:
                (x, y) = self.pos[node]
                center_x += x
                center_y += y
                count += 1
            center_x /= count
            center_y /= count
            aggregate_nodes[group] = {
                'center': (center_x, center_y),
                'count': count
            }
        aggregate_edges = {}
        for group_u in groups.keys():
            nodes_u = groups[group_u]
            for group_v in groups.keys():
                if group_u == group_v:
                    continue
                if (group_u, group_v) in aggregate_edges.keys() or \
                   (group_v, group_u) in aggregate_edges.keys():
                    continue
                count = 0
                nodes_v = groups[group_v]
                for u in nodes_u:
                    for v in nodes_v:
                        if self.G.has_edge(u, v):
                            count += 1
                if count == 0:
                    continue
                aggregate_edges[(group_u, group_v)] = {
                    'source': group_u,
                    'target': group_v,
                    'count': count}
        self.aggregations = {
            'nodes': aggregate_nodes,
            'edges': aggregate_edges
        }

    def draw_aggregations(
        self, size_min=5, size_max=2, width_min=2, width_max=1, agg_alpha=1
    ):
        ag_nodes = self.aggregations['nodes']
        ag_edges = self.aggregations['edges']
        self.ag_G = ag_G = nx.Graph()
        self.size_min = size_min
        self.size_max = size_max
        self.width_min = width_min
        self.width_max = width_max
        self.agg_alpha = agg_alpha
        pos = {}
        size = {}
        color = {}
        label = {}
        if self.node_weights is not None:
            node_weight_list = [(key, self.node_weights[key])
                                for key in self.node_weights.keys()]
            sorted_node_list = sorted(
                node_weight_list, key=lambda d: d[1], reverse=True)
            for (node, w) in sorted_node_list:
                if node.startswith("_") or node == "nan":
                    continue
                group = self.nodes[node][self.group_attribute]
                if group in label.keys():
                    continue
                label[group] = node
        for node in ag_nodes.keys():
            pos[node] = ag_nodes[node]['center']
            size[node] = ag_nodes[node]['count']
            color[node] = self.color_map[node]
            if label is {}:
                label[node] = node
            ag_G.add_node(node)
        for edge in ag_edges.keys():
            ag_G.add_edge(edge[0], edge[1])

        width_list = []
        for (s, t) in ag_G.edges:
            count = ag_edges[(s, t)]['count']
            width_list.append(count)
        width_list = normalize_list(width_list, self.width_min, self.width_max)
        width = {}
        for i, (s, t) in enumerate(ag_G.edges):
            width[(s, t)] = width_list[i]
        self.agg_width = width

        size_list = list(size.values())
        size_list = normalize_list(size_list, self.size_min, self.size_max)
        size = {}
        for i, node in enumerate(ag_nodes.keys()):
            size[node] = size_list[i]
        self.agg_size = size
        self.__compute_offsets()
        for node in ag_nodes.keys():
            pos[node] = (pos[node][0] + self.agg_offsets[node][0],
                         pos[node][1] + self.agg_offsets[node][1])

        color_list = list(color.values())

        nx.draw_networkx_nodes(
            ag_G, pos, node_size=size_list, node_color=color_list)
        # nx.draw_networkx_labels(ag_G, pos, labels=label, font_size=10)
        nx.draw_networkx_edges(ag_G, pos, width=width_list,
                               edge_color='gray', alpha=0.7)

    def set_events(self, fig):
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: self.__on_node_click(event))

    def select(self, select, fig):
        self.selected_list.append(select)
        self.ag_G.remove_node(select)
        self.__redraw(fig)
