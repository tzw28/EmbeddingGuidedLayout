import os
import json


def main():
    files = []
    contents = {}
    line_keys = []
    metrics = []
    path = './data/eval_results/2/'
    for graph in os.listdir(path):
        graph_path = os.path.join(path, graph)
        if not os.path.isdir(graph_path):
            continue
        if graph not in contents.keys():
            contents[graph] = {}
        for time in os.listdir(graph_path):
            time_path = os.path.join(graph_path, time)
            for file in os.listdir(time_path):
                if not file.endswith(".json"):
                    continue
                file_path = os.path.join(time_path, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                for method, results in content.items():
                    if method not in contents[graph].keys():
                        contents[graph][method] = {}
                    for metric, value in results.items():
                        line_keys.append((graph, method, metric))
                        if metric not in metrics:
                            metrics.append(metric)
                        contents[graph][method][metric] = value

    csv_path = "./data/eval_results/2/formatted.csv"
    header = ["graph", "method"] + metrics
    with open(csv_path, "w", ) as f:
        f.write(",".join(header) + "\n")
        for graph, graph_res in contents.items():
            for method, method_res in graph_res.items():
                line = [graph, method]
                for metric, value in method_res.items():
                    line.append(str(value))
                f.write(",".join(line) + "\n")


if __name__ == "__main__":
    main()
