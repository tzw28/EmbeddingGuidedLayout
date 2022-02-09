import bezier
import numpy as np


def left_of_line(p, p1, p2):
    tmpx = (p1[0] - p2[0]) / (p1[1] - p2[1]) * (p[1] - p2[1]) + p2[0]
    if tmpx > p[0]:
        return True
    return False


def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random', centers=None):
    # Get nodes into np array
    edges = np.array(list(G.edges()))
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l) == 0, -1, 1)
    elif polarity == 'graphlet' and centers is not None:
        rnd = []
        for edge in edges:
            vir_node = edge[0]
            vir_info = edge[0].split("_")
            det_node = edge[1]
            if len(vir_info) != 3:
                vir_node = edge[1]
                vir_info = edge[1].split("_")
                det_node = edge[0]
            center_node = vir_info[int(vir_info[2])]
            try:
                center = centers[int(center_node)]['center']
            except:
                center = centers[center_node]['center']
            if left_of_line(center, pos[vir_node], pos[det_node]):
                rnd.append(-1)
            else:
                rnd.append(1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(
            edges[:, 0])+np.vectorize(hash)(edges[:, 1]), 2) == 0, -1, 1)

    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse=True)
    coords = np.array([pos[x] for x in u])[inv].reshape(
        [edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:, 0, :]
    coords_node2 = coords[:, 1, :]

    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:, 0] > coords_node2[:, 0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]

    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:, 1]-coords_node1[:, 1]) / \
        (coords_node2[:, 0]-coords_node1[:, 0])
    m2 = -1/m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l), m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)), m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array(
        [coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:, i, :].T
        curveplots.append(bezier.Curve(nodes, degree=2).evaluate_multi(
            np.linspace(0, 1, bezier_precision)).T)

    # Return an array of these curves
    curves = np.array(curveplots)
    return curves
