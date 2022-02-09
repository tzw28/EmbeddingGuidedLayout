
from sklearn.manifold import TSNE


def tsne(G, vectors):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(vectors[key])
    nodes = list(G.nodes)
    tsne = TSNE(n_components=2)
    tsne.fit(vector_list)
    newX = tsne.fit_transform(vector_list)
    pos = {}
    for i in range(0, len(newX)):
        pos[nodes[i]] = newX[i]
    return pos
