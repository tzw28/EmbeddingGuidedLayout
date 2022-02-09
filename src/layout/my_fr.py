
from __future__ import division
import networkx as nx
import math
import random
import copy
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
import datetime
import matplotlib.pyplot as plt
import os
import time


def run_my_fr(G):
    layouter = MyFrLayouter()
    layouter.fr(G)
    layouter.normalize()
    return layouter.pos


class MyFrLayouter(object):
    def __init__(self, t=0.1, iters=1000, c=1, seed=8):

        self.t = t              # temperature
        self.iters = iters      # max interation number
        self.c = c              # distance control parameter
        self.area = 1           # layout in 1*1 space
        self.pos = {}           # result
        self.seed = seed        # seed
        self.threshold = 1e-5   # border

        self.path = self.make_figure_path(note="")

    '''
    Initialization
    '''

    def init_postions(self, G):
        random.seed(self.seed)
        for node in G.nodes:
            rand_x = round(random.uniform(0.0, 1.0), 4)
            rand_y = round(random.uniform(0.0, 1.0), 4)
            self.pos[node] = [rand_x, rand_y]

    '''
    Run fr layout iterations
    '''

    def fr(self, G):
        cur_t = self.t
        self.init_postions(G)
        for i in range(0, self.iters):
            cur_t = self.fr_iter(G, i, cur_t)
            if cur_t < self.threshold:
                print("----------break at ", i)
                break
        return copy.deepcopy(self.pos)

    '''
    Itration main function
    '''

    def fr_iter(self, G, cur_iter, cur_t):
        number_of_nodes = len(list(G.nodes))

        # equation for k
        k = self.c * math.sqrt(self.area / number_of_nodes)

        # Initialize displacement
        disp = {}
        for node in G.nodes:
            disp[node] = [0, 0]

        # repulsive force
        for u in G.nodes:
            for v in G.nodes:
                if u == v:
                    continue
                delta_x = self.pos[u][0] - self.pos[v][0]
                delta_y = self.pos[u][1] - self.pos[v][1]
                delta_len = math.sqrt(delta_x ** 2 + delta_y ** 2)
                delta_len = max(0.01, delta_len)
                force_r = k ** 2 / delta_len
                disp[u][0] += (delta_x / delta_len) * force_r
                disp[u][1] += (delta_y / delta_len) * force_r

        # attractive force
        for u, v in G.edges:
            delta_x = self.pos[u][0] - self.pos[v][0]
            delta_y = self.pos[u][1] - self.pos[v][1]
            delta_len = math.sqrt(delta_x ** 2 + delta_y ** 2)
            delta_len = max(0.01, delta_len)
            # force_a = delta_len ** 2 / k
            force_a = delta_len / k
            # a pair of force
            disp[u][0] -= (delta_x / delta_len) * force_a
            disp[u][1] -= (delta_y / delta_len) * force_a
            disp[v][0] += (delta_x / delta_len) * force_a
            disp[v][1] += (delta_y / delta_len) * force_a

        # update positon
        for u in G.nodes:
            disp_x = disp[u][0]
            disp_y = disp[u][1]
            disp_len = math.sqrt(disp_x ** 2 + disp_y ** 2)
            offset_x = disp_x * (cur_t / disp_len)
            offset_y = disp_y * (cur_t / disp_len)
            self.pos[u][0] = self.pos[u][0] + offset_x
            self.pos[u][1] = self.pos[u][1] + offset_y

        # cooling
        cur_t *= 1.0 - cur_iter / self.iters
        # self.save_temp(G, cur_iter)
        return cur_t

    def normalize(self):
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        for key in self.pos.keys():
            p = self.pos[key]
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]
            if p[1] < y_min:
                y_min = p[1]
            if p[1] > y_max:
                y_max = p[1]
        for key in self.pos.keys():
            p = self.pos[key]
            p[0] = (p[0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            p[1] = (p[1] - y_min) / (y_max - y_min) * 0.9 + 0.05

    def make_figure_path(self, note, err_msg=None):
        date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
        fig_path = "fig/" + date_path
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        time_str = time.strftime("%H-%M-%S", time.localtime(time.time()))
        save_path = fig_path + "/" + "miserables-temp"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += "/" + time_str
        if note:
            save_path += "-" + note
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if err_msg is not None:
            err_log_file = save_path + "/error_message.txt"
            with open(err_log_file, "w", encoding="utf-8") as f:
                f.write(err_msg)
        return save_path

    def save_temp(self, G, index):
        nx.draw(G, pos=self.pos, node_color='blue',
                node_size=10, width=0.1, edge_color='b')
        plt.savefig(self.path + "/{}.png".format(index))
        plt.cla()
