from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import LinearSVC


def kmeans(vectors, K=13):
    vector_list = list(vectors.values())
    if not K:
        SSE = []
        for k in range(2, 11):
            mod = KMeans(n_clusters=k, n_jobs=4, max_iter=500)
            mod.fit_predict(vector_list)
            SSE.append(mod.inertia_)
        max_delta = 0
        max_k = 0
        SSE_delta_list = []
        SSE_delta = 0
        for i, v in enumerate(SSE):
            if i == 0 or i == len(SSE) - 1:
                continue
            SSE_delta = SSE[i-1] + SSE[i+1] - 2*SSE[i]
            SSE_delta_list.append(SSE_delta)
            if SSE_delta > max_delta:
                max_delta = SSE_delta
                max_k = i + 2 + 2
        K = max_k

    mod = KMeans(n_clusters=K, n_jobs=1, max_iter=500, random_state=12)
    mod.fit_predict(vector_list)
    labels = mod.labels_
    groups = {}
    nodes = list(vectors.keys())
    for i in range(len(nodes)):
        node = nodes[i]
        groups[node] = labels[i]
    return groups


def agglomerative(vectors, K=13):
    vector_list = list(vectors.values())
    mod = AgglomerativeClustering(
        n_clusters=K, affinity="cosine", linkage="average")
    mod.fit_predict(vector_list)
    labels = mod.labels_
    groups = {}
    nodes = list(vectors.keys())
    for i in range(len(nodes)):
        node = nodes[i]
        groups[node] = labels[i]
    return groups


def svm(vectors):
    # 建立模型
    linear_svc = LinearSVC()
