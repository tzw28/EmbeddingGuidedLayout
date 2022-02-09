import os
import time


def run():
    for graph, params in param_table.items():
        for param_tuple in params:
            param_str = "_".join((str(i) for i in param_tuple))
            os.system("./Vis -input ./data/{g}_edge_list.txt -output ./pos/{g}_pos_{p}.txt -neg {neg} -samples {samples} -gamma {gamma} -mode {mode} -A {A} -B {B}".format(
                g=graph, p=param_str, neg=param_tuple[0], samples=param_tuple[1],
                gamma=param_tuple[2], mode=param_tuple[3], A=param_tuple[4],
                B=param_tuple[5],
            ))
            os.system(
                "python3 ./visualization/layout.py -graph ./data/{g}_edge_list.txt -layout ./pos/{g}_pos_{p}.txt -outpng ./pic/{g}_{p}.png".format(g=graph, p=param_str))


# (neg, samples, gamma, mode, A, B)
param_table = {
    "miserables": [
        (5, 1, 0.1, 1, 2, 1),
        (5, 1, 0.01, 1, 2, 1),
        (5, 1, 0.5, 1, 2, 1),
        (5, 10, 0.1, 1, 2, 1),
        (5, 10, 0.01, 1, 2, 1),
        (5, 10, 0.5, 1, 2, 1),
    ],
    "science": [
        (5, 1, 0.1, 1, 2, 1),
        (5, 1, 0.01, 1, 2, 1),
        (5, 1, 0.5, 1, 2, 1),
        (5, 10, 0.1, 1, 2, 1),
        (5, 10, 0.01, 1, 2, 1),
        (5, 10, 0.5, 1, 2, 1),
        (5, 100, 0.1, 1, 2, 1),
        (5, 100, 0.01, 1, 2, 1),
        (5, 100, 0.5, 1, 2, 1),
    ],
    "facebook": [
        (5, 1, 0.1, 1, 2, 1),
        (5, 1, 0.01, 1, 2, 1),
        (5, 1, 0.5, 1, 2, 1),
        (5, 10, 0.1, 1, 2, 1),
        (5, 10, 0.01, 1, 2, 1),
        (5, 10, 0.5, 1, 2, 1),
        (5, 100, 0.1, 1, 2, 1),
        (5, 100, 0.01, 1, 2, 1),
        (5, 100, 0.5, 1, 2, 1),
    ],
    "cora": [
        (5, 1, 0.1, 1, 2, 1),
        (5, 1, 0.01, 1, 2, 1),
        (5, 1, 0.5, 1, 2, 1),
        (5, 10, 0.1, 1, 2, 1),
        (5, 10, 0.01, 1, 2, 1),
        (5, 10, 0.5, 1, 2, 1),
        (5, 100, 0.1, 1, 2, 1),
        (5, 100, 0.01, 1, 2, 1),
        (5, 100, 0.5, 1, 2, 1),
        (5, 500, 0.1, 1, 2, 1),
        (5, 500, 0.01, 1, 2, 1),
        (5, 500, 0.5, 1, 2, 1),
    ],
    "citeseer": [
        (5, 1, 0.1, 1, 2, 1),
        (5, 1, 0.01, 1, 2, 1),
        (5, 1, 0.5, 1, 2, 1),
        (5, 10, 0.1, 1, 2, 1),
        (5, 10, 0.01, 1, 2, 1),
        (5, 10, 0.5, 1, 2, 1),
        (5, 100, 0.1, 1, 2, 1),
        (5, 100, 0.01, 1, 2, 1),
        (5, 100, 0.5, 1, 2, 1),
        (5, 500, 0.1, 1, 2, 1),
        (5, 500, 0.01, 1, 2, 1),
        (5, 500, 0.5, 1, 2, 1),
    ]
}


if __name__ == "__main__":
    run()
