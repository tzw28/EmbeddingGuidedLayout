import networkx as nx
import math
import numpy as np


def nx_fr(G):
    # return nx.fruchterman_reingold_layout(G, center=(0.5, 0.5), seed=17)
    pos = nx.fruchterman_reingold_layout(G, seed=17)
    return pos
