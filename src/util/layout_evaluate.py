import networkx as nx
import os
from src.layout.evaluator import LayoutEvaluator


def load_dr_graph(graph):
    graph_path = "./data/dr_examples/" + graph + ".txt"
    pos_path = "./data/dr_positions/" + graph + ".txt_pos.txt"
    G = nx.Graph()
    with open(graph_path, "r") as f:
        line = f.readline()
        node_number = int(line.split()[0])
        edge_number = int(line.split()[1])
        line = f.readline()
        while line:
            n1 = int(line.split()[0])
            n2 = int(line.split()[1])
            G.add_node(n1)
            G.add_node(n2)
            G.add_edge(n1, n2)
            line = f.readline()
    pos = {}
    with open(pos_path, "r") as f:
        line = f.readline()
        node_number = int(line.split()[0])
        line = f.readline()
        index = 0
        while line:
            x = float(line.split()[0])
            y = float(line.split()[1])
            pos[index] = [x, y]
            index += 1
            line = f.readline()
    return G, pos


def eval_dr_graphs(G, pos, graph_name):
    eva = LayoutEvaluator(G, pos, None)
    eva.run()
    eva.save_json_result("./data/dr_examples/", graph_name)


def run_layout_evaluation():
    dr_graphs = []
    for file in os.listdir("./data/dr_examples"):
        if not file.endswith(".txt"):
            continue
        graph = file.split(".txt")[0]
        dr_graphs.append(graph)
    pos_dict = {}
    for graph in dr_graphs:
        if os.path.exists("./data/dr_examples/" + graph + "_layout_evaluation.json"):
            continue
        G, pos = load_dr_graph(graph)
        pos_dict[graph] = pos
        try:
            eval_dr_graphs(G, pos_dict, graph)
        except:
            continue
