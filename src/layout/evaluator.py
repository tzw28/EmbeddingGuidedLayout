from matplotlib.pyplot import contour
from src.util.graph_reading import largest_connected_subgraph
import numpy as np
import json
import math
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

from numpy.core.numeric import cross
import matplotlib.path as mpltPath


ZERO = 1e-9


# 点
class Point(object):

    def __init__(self, x, y):
        self.x, self.y = x, y


# 向量
class Vector(object):

    def __init__(self, start_point, end_point):
        self.start_point, self.end_point = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y

    def negative(self):
        return Vector(self.end_point, self.start_point)


class LayoutEvaluator(object):

    def __init__(self, graph_name, G, pos_dict, class_map, limit_to=None):
        self.graph_name = graph_name
        self.G = G
        self.pos_dict = pos_dict
        self.class_map = class_map
        self.dis_map = {}
        self.limit_to = limit_to

    def _normalize_position(self, pos, x_range=(0, 1), y_range=(0, 1)):
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

    def _distance(self, n1, n2, pos):
        pos1 = np.array(pos[n1])
        pos2 = np.array(pos[n2])
        dis = np.linalg.norm(pos1 - pos2)
        return dis

    def _vector_product(self, vector1, vector2):
        return vector1.x * vector2.y - vector2.x * vector1.y

    def _is_intersected(self, A, B, C, D):
        if min(A[0], B[0]) > max(C[0], D[0]) or \
           min(A[1], B[1]) > max(C[1], D[1]) or \
           min(C[0], D[0]) > max(A[0], B[0]) or \
           min(C[1], D[1]) > max(A[1], B[1]):
            return False
        PA = Point(A[0], A[1])
        PB = Point(B[0], B[1])
        PC = Point(C[0], C[1])
        PD = Point(D[0], D[1])
        VAC = Vector(PA, PC)
        VAD = Vector(PA, PD)
        VBC = Vector(PB, PC)
        VBD = Vector(PB, PD)
        VCA = VAC.negative()
        VCB = VBC.negative()
        VDA = VAD.negative()
        VDB = VBD.negative()
        temp1 = self._vector_product(VAC, VAD) * self._vector_product(VBC, VBD)
        temp2 = self._vector_product(VCA, VCB) * self._vector_product(VDA, VDB)
        return (temp1 <= ZERO) and (temp2 <= ZERO)

    def _compute_edge_lengths(self, pos):
        lens = []
        for s, t in self.G.edges:
            dis = self._distance(s, t, pos)
            lens.append(dis)
        return lens

    def _compute_edge_length_uniformity(self, pos):
        edge_lengths = self._compute_edge_lengths(pos)
        len_arr = np.std(edge_lengths, ddof=1)
        len_mean = np.mean(edge_lengths)
        uni = len_arr / len_mean
        return uni

    def _compute_node_distribution(self, pos):
        lens = self._compute_edge_lengths(pos)
        res = 0
        for l in lens:
            res += 1 / l**2
        return res

    def _compute_edge_crossings(self, pos, outside=False):
        if outside and not self.class_map:
            return -1
        crossing_count = 0
        edge_num = len(self.G.edges)
        total_degree = 0
        for node in self.G.nodes:
            degree = self.G.degree(node)
            total_degree += degree
        # print(edge_num)
        for s1, t1 in self.G.edges:
            if outside and self.class_map[s1] == self.class_map[t1]:
                continue
            for s2, t2 in self.G.edges:
                if outside and self.class_map[s2] == self.class_map[t2]:
                    continue
                if s1 == t2 or s1 == s2 or s2 == t1 or \
                   t1 == s2 or t1 == t2 or t2 == s1:
                    # print("skip {}-{} {}-{}".format(s1, t1, s2, t2))
                    continue
                ps1, pt1 = pos[s1], pos[t1]
                ps2, pt2 = pos[s2], pos[t2]
                if self._is_intersected(ps1, pt1, ps2, pt2):
                    # print("cross {}-{} {}-{}".format(s1, t1, s2, t2))
                    crossing_count += 1
        # if not outside:
            # print("crossings {}".format(crossing_count))
        # return crossing_count / 2 / edge_num ** 2
        return crossing_count / edge_num ** 2

    def _compute_community_significance(self, pos):
        if not self.class_map:
            return -1
        inner_dis = []
        outer_dis = []
        for node1 in self.G.nodes:
            for node2 in self.G.nodes:
                dis = self._distance(node1, node2, pos)
                if self.class_map[node1] == self.class_map[node2]:
                    inner_dis.append(dis)
                else:
                    outer_dis.append(dis)
        return np.mean(inner_dis), np.mean(outer_dis)

    def _knbrs(self, G, start, k):
        nbrs = set([start])
        for l in range(k):
            nbrs = set((nbr for n in nbrs for nbr in G[n]))
        return nbrs

    def _distance_matrix(self, pos):
        if self.dis_map:
            # print("distance map exists.")
            return
        dis_map = {}
        # distance matrix
        pos_list = list(pos.values())
        node_key_list = list(pos.keys())
        distance_matrix = cdist(
            pos_list, pos_list, 'euclidean')  # euclidean表示欧式距离
        for node1 in self.G.nodes:
            if node1 not in dis_map.keys():
                dis_map[node1] = {}
            for node2 in self.G.nodes:
                if node1 == node2:
                    dis = 0
                else:
                    index1 = node_key_list.index(node1)
                    index2 = node_key_list.index(node2)
                    dis = distance_matrix[index1][index2]
                dis_map[node1][node2] = dis
        self.dis_map = dis_map

    def _compute_normalized_distance_matrix(self):
        if not self.dis_map:
            print("No distance map.")
            return None
        min_dis = 99999
        max_dis = 0
        for node1 in self.dis_map.keys():
            for node2 in self.dis_map.keys():
                if node1 == node2:
                    continue
                dis = self.dis_map[node1][node2]
                if dis > max_dis:
                    max_dis = dis
                if dis < min_dis:
                    min_dis = dis
        
        nml_dis_map = {}
        for node1 in self.dis_map.keys():
            nml_dis_map[node1] = {}
            for node2 in self.dis_map.keys():
                if node1 == node2:
                    nml_dis_map[node1][node2] = 0
                    continue
                dis = self.dis_map[node1][node2]
                nml_dis_map[node1][node2] = (dis - min_dis) / (max_dis - min_dis)
        return nml_dis_map


    def _compute_neighborhood_preservation(self, pos, k=2):
        dis_map = {}
        # distance matrix
        self._distance_matrix(pos)
        dis_map = self.dis_map
        # compute np
        res = 0
        for node in self.G.nodes:
            k_nbrs = self._knbrs(self.G, node, k)
            ki = len(k_nbrs)
            nearest_ki = set(
                sorted(dis_map[node], key=lambda x: dis_map[node][x])[:ki])
            if len(k_nbrs | nearest_ki) == 0:
                # print(k_nbrs, nearest_ki)
                continue
            res += len(k_nbrs & nearest_ki) / len(k_nbrs | nearest_ki)
        res /= self.G.number_of_nodes()
        return res

    def _compute_crosslessness(self, pos):
        # c_max
        edge_num = len(self.G.edges)
        cmax = edge_num * (edge_num - 1) * 0.5
        for node, deg in self.G.degree():
            cmax -= deg * (deg - 1) * 0.5
        # crossing number
        crossing_count = 0
        total_degree = 0
        for node in self.G.nodes:
            degree = self.G.degree(node)
            total_degree += degree
        # print(edge_num)
        exmained_edge_pair = {}
        edge_list = list(self.G.edges)
        for i in range(0, len(edge_list)):
            s1, t1 = edge_list[i]
            for j in range(i + 1, len(edge_list)):
                s2, t2 = edge_list[j]
                if s1 == t2 or s1 == s2 or s2 == t1 or t1 == t2:
                    # print("skip {}-{} {}-{}".format(s1, t1, s2, t2))
                    continue
                ps1, pt1 = pos[s1], pos[t1]
                ps2, pt2 = pos[s2], pos[t2]
                if self._is_intersected(ps1, pt1, ps2, pt2):
                    # print("cross {}-{} {}-{}".format(s1, t1, s2, t2))
                    crossing_count += 1

        # for s1, t1 in self.G.edges:
        #     for s2, t2 in self.G.edges:
        #         if s1 == t2 or s1 == s2 or s2 == t1 or t1 == t2:
        #             # print("skip {}-{} {}-{}".format(s1, t1, s2, t2))
        #             continue
        #         ps1, pt1 = pos[s1], pos[t1]
        #         ps2, pt2 = pos[s2], pos[t2]
        #         if self._is_intersected(ps1, pt1, ps2, pt2):
        #             # print("cross {}-{} {}-{}".format(s1, t1, s2, t2))
        #             crossing_count += 1
        # print("crossings {}".format(crossing_count))
        if cmax > 0:
            cln = 1 - math.sqrt(crossing_count / cmax)
        else:
            cln = 1
        return cln

    def _compute_minimum_angle(self, pos):
        # distance matrix
        self._distance_matrix(pos)
        dis_map = self.dis_map
        # minimum angle
        temp = 0
        for node in self.G.nodes:
            nbrs = self._knbrs(self.G, node, 1)
            if len(nbrs) < 2:
                continue
            min_angle = 180
            for node_a in nbrs:
                for node_b in nbrs:
                    if node_a == node_b:
                        continue
                    oa = dis_map[node][node_a]
                    ob = dis_map[node][node_b]
                    ab = dis_map[node_a][node_b]
                    cos_o = (oa * oa + ob * ob - ab * ab) / (2 * oa * ob)
                    if cos_o > 1 or cos_o < -1:
                        print("bad cos value")
                        continue
                    angle_o = math.acos(cos_o) / math.pi * 180
                    if angle_o < min_angle:
                        min_angle = angle_o
            ideal_angle = 360 / len(nbrs)
            temp += abs(ideal_angle - min_angle) / ideal_angle
        ma = 1 - temp / self.G.number_of_nodes()
        return ma

    def _compute_node_spread(self, pos):
        if not self.class_map:
            return -1
        centers = {}
        for node in self.G.nodes:
            clas = self.class_map[node]
            if clas not in centers.keys():
                # ceneterx, centery, number, distance
                centers[clas] = [0, 0, 0, 0]
            centers[clas][0] += pos[node][0]
            centers[clas][1] += pos[node][1]
            centers[clas][2] += 1
        for clas in centers.keys():
            centers[clas][0] /= centers[clas][2]
            centers[clas][1] /= centers[clas][2]
        for node in self.G.nodes:
            clas = self.class_map[node]
            pos_x = pos[node][0]
            pos_y = pos[node][1]
            dis = math.sqrt(
                (pos_x - centers[clas][0]) ** 2 + (pos_y - centers[clas][1]) ** 2)
            centers[clas][3] += dis
        res = 0
        for clas in centers.keys():
            avg_dis = centers[clas][3] / centers[clas][2]
            res += avg_dis
        res /= len(centers.keys())
        return res

    def _distance_to_hull(self, target, convex_hull):
        l = convex_hull.shape[0]

        def obj(x):
            result = target - sum(np.dot(np.diag(x), convex_hull))
            return np.linalg.norm(result)

        # 不等式约束
        ineq_cons = {"type": "ineq",
                     "fun": lambda x: x}

        # 等式约束
        eq_cons = {"type": "eq",
                   "fun": lambda x: sum(x)-1}

        x0 = np.ones(l)/l

        res = minimize(obj, x0, method='SLSQP', constraints=[
                       eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': False})

        return res.fun

    def _in_convexhull(self, p, path):
        return path.contains_point(p)

    def _compute_group_overlap(self, pos):
        if not self.class_map:
            return -1
        group_nodes = {}
        group_hulls = {}
        for node in self.G.nodes:
            clas = self.class_map[node]
            if clas not in group_nodes.keys():
                group_nodes[clas] = []
            group_nodes[clas].append(pos[node])
        for clas, pos_list in group_nodes.items():
            if len(pos_list) < 3:
                group_hulls[clas] = None
                continue
            hull = ConvexHull(pos_list)
            group_hulls[clas] = hull
        res = 0
        group_number = 0
        for group, hull in group_hulls.items():
            hull_count = 0
            if hull is None:
                continue
            group_number += 1
            node_number = 0
            path = mpltPath.Path(hull.points)
            for node in self.G.nodes:
                clas = self.class_map[node]
                p = pos[node]
                if clas == group:
                    continue
                node_number += 1
                # dis = self._distance_to_hull(p, hull.points)
                # if dis <= 1e-05:
                #     hull_count += 1
                if self._in_convexhull(p, path):
                    hull_count += 1
            res += hull_count / node_number
        return res / group_number

    def _compute_global_entropy(self, pos, unit_width=0.2):
        if not self.class_map:
            return -1
        class_count = 0
        temp = []
        for node, clas in self.class_map.items():
            if clas in temp:
                continue
            temp.append(clas)
            class_count += 1
        unit_class_dict = {}
        miss_count = 0
        for node, p in pos.items():
            if node not in self.class_map.keys():
                miss_count += 1
                continue
            clas = self.class_map[node]
            index_x = int(p[0] / unit_width)
            index_y = int(p[1] / unit_width)
            unit = (index_x, index_y)
            if unit not in unit_class_dict.keys():
                unit_class_dict[unit] = [clas]
            else:
                unit_class_dict[unit].append(clas)
        if miss_count > 0:
            print("missed {}".format(miss_count))
        res = 0
        unit_count = 0
        entropy = {}
        for unit, classes in unit_class_dict.items():
            class_dict = {}
            count = 0
            ent = 0
            for clas in classes:
                count += 1
                if clas not in class_dict.keys():
                    class_dict[clas] = 1
                else:
                    class_dict[clas] += 1
            for clas, clas_count in class_dict.items():
                class_percentage = clas_count / count
                ent += -1 * class_percentage * math.log2(class_percentage)
            # ent /= class_count # 归一化？好像不太可能
            entropy[unit] = ent
            unit_count += 1
            res += ent
        res /= unit_count
        return res
    
    def _compute_autocorrelation(self, pos, radius=0.1):
        if not self.class_map:
            return -1
        normalized_dis_map = self._compute_normalized_distance_matrix()
        res = 0
        node_count = 0
        for nodei in pos.keys():
            if nodei not in normalized_dis_map.keys():
                continue
            node_count += 1
            count = 0
            A = 0
            B = 0
            for nodej in pos.keys():
                if nodei == nodej or nodej not in normalized_dis_map.keys():
                    continue
                dis = normalized_dis_map[nodei][nodej]
                if dis > radius:
                    continue
                count += 1
                B += 1 - dis
                if self.class_map[nodei] != self.class_map[nodej]:
                    A += 1 - dis
            if B == 0:
                continue
            Ci = A / B
            res += Ci
        res /= node_count
        return res
    
    def _compute_node_occlusions(self, pos, threshold=0.01):
        count = 0
        node_count = 0
        for node1 in pos.keys():
            if node1 not in self.dis_map.keys():
                continue
            node_count += 1
            for node2 in pos.keys():
                if node1 == node2 or node2 not in self.dis_map.keys():
                    continue
                dis = self.dis_map[node1][node2]
                if dis < threshold:
                    count += 1
        res = count / 2 / node_count ** 2
        return res

    def run(self):
        print("开始布局效果评估计算")
        self.res_dict = {}
        for key in self.pos_dict.keys():
            if self.limit_to and key not in self.limit_to:
                continue
            print(key)
            G = self.G
            if key == "PH":
                self.G = largest_connected_subgraph(G)
            pos = self._normalize_position(self.pos_dict[key])
            self.dis_map = {}
            self._distance_matrix(pos)
            # inner_avg_distance, outer_avg_distance = self._compute_community_significance(pos)
            self.res_dict[key] = {
                # "edge_length_uniformity": self._compute_edge_length_uniformity(pos),
                # "node_distribution": self._compute_node_distribution(pos),
                # "edge_crossings": self._compute_edge_crossings(pos),
                # "np-1": self._compute_neighborhood_preservation(pos, k=1),
                # "np-2": self._compute_neighborhood_preservation(pos, k=2),
                # "np-3": self._compute_neighborhood_preservation(pos, k=3),
                # "np-4": self._compute_neighborhood_preservation(pos, k=4),
                # "np-5": self._compute_neighborhood_preservation(pos, k=5),
                # "cln": self._compute_crosslessness(pos),

                "node_spread": self._compute_node_spread(pos),
                "node_occlusions": self._compute_node_occlusions(pos),
                "edge_crossings": self._compute_edge_crossings(pos),
                "edge_crossings_outside": self._compute_edge_crossings(pos, outside=True),
                "minimum_angle": self._compute_minimum_angle(pos),
                "edge_length_variation": self._compute_edge_length_uniformity(pos),
                "group_overlap": self._compute_group_overlap(pos),
                "entropy": self._compute_global_entropy(pos),
                "autocorrelation": self._compute_autocorrelation(pos)
            }
            self.G = G

    def save_json_result(self, save_path):
        file_path = save_path + "/{}_layout_evaluation.json".format(self.graph_name)
        with open(file_path, "w") as f:
            json.dump(self.res_dict, f)
        print("布局效果评估计算完成")
