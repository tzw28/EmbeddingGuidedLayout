import networkx as nx
from networkx.utils import random_state
import math
import numba
import matplotlib.pyplot as plt
import json
from src.util.graph_reading import clean_attributed_graph
import numpy as np


g_colors = []


def _normalize_positions(pos):
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for key in pos.keys():
        p = pos[key]
        if p[0] < x_min:
            x_min = p[0]
        if p[0] > x_max:
            x_max = p[0]
        if p[1] < y_min:
            y_min = p[1]
        if p[1] > y_max:
            y_max = p[1]
    for key in pos.keys():
        p = pos[key]
        p[0] = (p[0] - x_min) / (x_max - x_min) * 0.9 + 0.05
        p[1] = (p[1] - y_min) / (y_max - y_min) * 0.9 + 0.05


def embedding_fr(G, vectors, pos=None, te=0.6, wa=1, we=1,
                 dis_method="euclidean", cluster=None, fixed=None,
                 tel=0.6, teh=0.85):
    clean_attributed_graph(G, vectors=vectors)
    pos = attr_fruchterman_reingold_layout(
        G, vectors=vectors, pos=pos, seed=(17), te=te, wa=wa, we=we,
        dis_method=dis_method, cluster=cluster, fixed=fixed,
        tel=tel, teh=teh
    )
    _normalize_positions(pos)
    return pos


def read_colors(G, File_name):
    node_groups = {}
    with open(File_name, "r") as f:
        text = f.read()
        json_graph = json.loads(text)
        nodes = json_graph["nodes"]
        for node in nodes:
            node_groups[node["id"].replace(" ", "")] = node["group"]
    gt_color_list = []
    nodes = list(G.nodes)
    for node in nodes:
        gt_color_list.append(node_groups[node])
    global g_colors
    g_colors = gt_color_list.copy()


def _distance(u, v, vectors, method):
    from scipy.spatial import distance
    method_map = {
        "braycurtis": distance.braycurtis,
        "canberra": distance.canberra,
        "chebyshev": distance.chebyshev,
        "cityblock": distance.cityblock,
        "correlation": distance.correlation,
        "cosine": distance.cosine,
        "euclidean": distance.euclidean,
        "jensenshannon": distance.jensenshannon,
        "mahalanobis": distance.mahalanobis,
        "minkowski": distance.minkowski,
        "seuclidean": distance.seuclidean,
        "sqeuclidean": distance.sqeuclidean,
    }

    u_vec = vectors[u]
    v_vec = vectors[v]
    # dis = distance.cosine(u_vec, v_vec)
    dis = method_map[method](u_vec, v_vec)
    return dis


@numba.jit(nopython=True)
def __cosine_dis(vec1, vec2):
    vector_a = np.mat(vec1)
    vector_b = np.mat(vec2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


@numba.jit(nopython=True)
def __euclidean_distance(x1, x2):
    return -2*np.dot(x1, x2.T) + np.expand_dims(np.sum(np.square(x1), axis=1), axis=1) + np.sum(np.square(x2), axis=1)
    # return np.sqrt(sum(np.power((vec1 - vec2), 2)))


@numba.jit(nopython=True)
def euclidean_distance_square_numba_v3(x1, x2):
    res = np.empty(x2.shape[0], dtype=x2.dtype)
    val = 0
    for idx in range(x2.shape[0]):
        tmp = x1[idx] - x2[idx]
        val += tmp * tmp
    res = np.sqrt(val)
    return res

# @numba.jit(nopython=True)


def __fast_sim_matrix(vec_list, size):
    sim_mat = np.empty((size, size), dtype=np.float32)
    for i in range(0, size):
        for j in range(0, size):
            sim_mat[i][j] = euclidean_distance_square_numba_v3(
                vec_list[i], vec_list[j])
            # sim_mat[i][j] = euclidean_distance_square_einsum(vec_list[i], vec_list[j])
    return sim_mat


def _process_vectors(vectors, dis_method="cosine"):
    import numpy as np
    vec_list = []
    for u in vectors.keys():
        vec_list.append(vectors[u])
    size = len(vec_list)
    E1 = __fast_sim_matrix(vec_list, size)
    return E1
    E = []
    # print("Using method {} to compute the distances.".format(dis_method))
    for u in vectors.keys():
        similarity = []
        for v in vectors.keys():
            if u == v:
                similarity.append(1)
                continue
            d = _distance(u, v, vectors, dis_method)
            similarity.append(d)
        E.append(similarity)
    return np.array(E)


def _normalize_mat(D):
    d_max = D.max()
    d_min = D.min()
    D1 = (D - d_min) / (d_max - d_min)
    return D1


def _embed_adjacency_matrix(A, E, wa, we, te, cluster_list=None,
                            tel=0.6, teh=0.85):
    if E is not None:
        D1 = _normalize_mat(E)
        D1 = 1 - D1
        D2 = wa * A + we * D1
        # D2 = _normalize_mat(D2)
        # D1[D1 <= te] = 0
        # D2 = wa * A + we * D1
        if cluster_list is None:
            D2[D2 <= te] = 0
        else:
            for i in range(len(cluster_list)):
                for j in range(len(cluster_list)):
                    c_i = cluster_list[i]
                    c_j = cluster_list[j]
                    if c_i == c_j:
                        te = tel
                    else:
                        te = teh
                    D2[i][j] = 0 if D2[i][j] < te else D2[i][j]
        A = _normalize_mat(D2)
    return A


def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


@random_state(11)
def attr_fruchterman_reingold_layout(
    G,
    k=None,
    pos=None,
    vectors=None,
    fixed=None,
    iterations=100,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    wa=1,
    we=1,
    te=0.6,
    dis_method="euclidean",
    cluster=None,
    tel=0.6,
    teh=0.85
):
    """Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.

    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.
        ValueError raised if `fixed` specified and `pos` not.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)

    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    E = None
    if vectors is not None:
        E = _process_vectors(vectors, dis_method=dis_method)

    # read_colorsread_colors(G, "./mis.json")

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    cluster_list = None
    if cluster:
        cluster_list = []
        for node in list(G.nodes):
            cluster_list.append(cluster[node])
    try:
        # Sparse matrix
        if len(G) > 300:  # sparse solver for large graphs
            raise ValueError
        A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype="f")
        A = _embed_adjacency_matrix(A, E, wa, we, te,
                                    cluster_list=cluster_list, tel=tel, teh=teh)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    except:
        A = nx.to_numpy_array(G, weight=weight)
        A = _embed_adjacency_matrix(A, E, wa, we, te,
                                    cluster_list=cluster_list, tel=tel, teh=teh)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    if fixed is None and scale is not None:
        pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


@random_state(7)
def _fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg)

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k)
        )
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


@random_state(7)
def _sparse_fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg)
    try:
        from scipy.sparse import spdiags, coo_matrix
    except ImportError:
        msg = "_sparse_fruchterman_reingold() scipy numpy: http://scipy.org/ "
        raise ImportError(msg)
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except:
        A = (coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta ** 2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            displacement[:, i] += (
                delta * (k * k / distance ** 2 - Ai * distance / k)
            ).sum(axis=1)
        # update positions
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos
