import json
import networkx as nx
import random


class JsonMetaPathGenerator:
    def __init__(self):
        pass

    def read_data(self, dirpath, filename):
        G = nx.Graph()
        with open(dirpath + "/" + filename, encoding="utf-8") as f:
            text = f.read()
            json_graph = json.loads(text)
            nodes = json_graph["nodes"]
            links = json_graph["links"]
            for node in nodes:
                attr_dict = {}
                for key in node.keys():
                    if key not in ["x", "y", "id"]:
                        attr_dict[key] = node[key]
                node_name = node["id"].replace(" ", "")
                node_attrs = attr_dict
                G.add_node(node_name, **node_attrs)
                # G.add_node(node_name)
            for link in links:
                G.add_edge(link["source"].replace(" ", ""),
                           link["target"].replace(" ", ""))
        self.G = G

    def generate_random_aca(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        random.seed(1729)
        nodes_list = list(self.G.nodes)
        for node in nodes_list:
            cur_node = node
            for j in range(0, numwalks):  # wnum walks
                outline = cur_node
                for i in range(0, walklength):
                    numa = len(nodes_list)
                    nodeid = random.randrange(numa)
                    newnode = nodes_list[nodeid]
                    while (not self.G.has_edge(newnode, cur_node) and not self.G.has_edge(cur_node, newnode)):
                        nodeid = random.randrange(numa)
                        newnode = nodes_list[nodeid]
                    outline += " " + newnode
                    cur_node = newnode
                outfile.write(outline + "\n")
        outfile.close()

    def generate_node_type_map(self, path_file, type_map_file):
        type_map = {}
        with open(path_file, 'r') as f:
            for line in f:
                toks = line.strip().split(" ")
                for node in toks:
                    type_map[node] = 'character'
        with open(type_map_file, 'w') as f:
            for key in type_map.keys():
                outline = "{} {}".format(key, type_map[key])
                f.write(outline + "\n")

    def generate_nx_graph(self, vectors):
        for node in self.G.nodes:
            if node not in vectors.keys():
                self.G.remove_node(node)
        print(len(self.G.nodes))
        print(len(vectors))
        return self.G

    def get_real_types(self):
        real_types = {}
        for node in self.G.nodes(data=True):
            node_name = node[0]
            attrs = node[1]
            real_types[node_name] = attrs['group']
        return real_types
