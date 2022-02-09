import os
import time


def run():
    data_dir = "./data"
    pos_dir = "./pos"
    pic_dir = "./pic"
    param_tuple = (5, 400, 0.1, 1, 2, 1)
    for graph in os.listdir(data_dir):
        graph_path = os.path.join(data_dir, graph)
        pos_path = os.path.join(pos_dir, graph + "_pos.txt")
        pic_path = os.path.join(pic_dir, graph + "_pic.png")
        os.system("./Vis -input {g} -output {p} -neg {neg} -samples {samples} -gamma {gamma} -mode {mode} -A {A} -B {B}".format(
            g=graph_path, p=pos_path, neg=param_tuple[0], samples=param_tuple[1],
            gamma=param_tuple[2], mode=param_tuple[3], A=param_tuple[4],
            B=param_tuple[5],
        ))
        os.system("python3 ./visualization/layout.py -graph {g} -layout {p} -outpng {pic}".format(
            g=graph_path, p=pos_path, pic=pic_path))


if __name__ == "__main__":
    run()
