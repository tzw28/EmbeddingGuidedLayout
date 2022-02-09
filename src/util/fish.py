import copy
import numpy as np
import networkx as nx

# AutoVis图数据graph中边的起点、终点名称
source_name = "source"
target_name = "target"

"""
获取初始布局
通过调用networkx库的布局算法实现。
输入c是核心点index，int类型数据。graph是需要布局的图，AutoVis图数据。type是布局类型，对应关系如下：
	布局	Type
	circular	1
	fruchterman_reingold布局1	9
	fruchterman_reingold布局2	9.2（默认布局）
	kamada_kawai布局1	2（效果最优但耗时）
	kamada_kawai布局2	2.1（考虑点的权重，点需要有weight属性）
	random	3
	spectral	7
无返回值，输入graph的nodes的x和y会被修改成布局中的点，且大小缩放到[0,1]。
"""


def getStartLayout(c, graph, type=9.2):  # pragma: no cover
    # 将AutoVis图转化为networkx图
    nxG = nx.Graph()
    nodes = graph["nodes"]
    nxG.add_nodes_from(range(len(nodes)))
    links = graph["links"]
    for i in range(len(links)):
        if(links[i][source_name] != links[i][target_name]):
            nxG.add_edge(links[i][source_name], links[i][target_name])

    # 调用nx布局算法获得初始布局
    if(type == 0):
        pos = nx.bipartite_layout(nxG, center=(0.5, 0.5), nodes=[c])
    elif(type == 0.1):
        sub = []
        for i in range(len(nodes)):
            if(nodes[i]["weight"] >= 0.5):
                sub.append(i)
        pos = nx.bipartite_layout(nxG, center=(0.5, 0.5), nodes=sub)
    elif(type == 1):
        pos = nx.circular_layout(nxG, center=(0.5, 0.5))
    elif(type == 2):
        pos = nx.kamada_kawai_layout(nxG, center=(0.5, 0.5))
    elif(type == 2.1):
        dist = {c: {}}
        for i in range(len(nodes)):
            dist[c][i] = 1-nodes[i]["weight"]
        pos = nx.kamada_kawai_layout(nxG, center=(0.5, 0.5), dist=dist)
    elif(type == 2.2):
        dist = {c: {}}
        for i in range(len(nodes)):
            d = 1-nodes[i]["weight"]
            m = 3
            dist[c][i] = (m + 1.) * d / (m * d + 1.)
        pos = nx.kamada_kawai_layout(nxG, center=(0.5, 0.5), dist=dist)
    elif(type == 2.3):
        p = {}
        for i in range(len(nodes)):
            p[i] = np.array([nodes[i]["weight"], nodes[i]["weight"]])
        pos = nx.kamada_kawai_layout(nxG, center=(0.5, 0.5), pos=p)
    elif(type == 3):
        pos = nx.random_layout(nxG, center=(0.5, 0.5), seed=17)
    # elif(type == 4):
        #pos = nx.rescale_layout(nxG)
    elif(type == 5):
        pos = nx.shell_layout(nxG, center=(0.5, 0.5))
    elif(type == 6):
        pos = nx.spring_layout(
            nxG, pos={c: np.array([0., 0.])}, fixed=[c], seed=17)
    elif(type == 7):
        pos = nx.spectral_layout(nxG, center=(0.5, 0.5))
    elif(type == 8):
        pos = nx.planar_layout(nxG, center=(0.5, 0.5))
    elif(type == 9):
        pos = nx.fruchterman_reingold_layout(
            nxG, pos={c: np.array([0.5, 0.5])}, fixed=[c], seed=17)
    elif(type == 9.1):
        p = {}
        for i in range(len(nodes)):
            p[i] = np.array([nodes[i]["weight"]/2., nodes[i]["weight"]/2.])
        pos = nx.fruchterman_reingold_layout(nxG, pos=p, fixed=[c], seed=17)
    elif(type == 9.2):
        pos = nx.fruchterman_reingold_layout(nxG, center=(0.5, 0.5), seed=17)

    # 调整初始布局，使x，y取值限制在[0,1]之中，并将布局后点的位置写入graph
    maxx = -np.inf
    maxy = -np.inf
    minx = np.inf
    miny = np.inf
    for i in range(len(nodes)):
        maxx = max(maxx, pos[i][0])
        maxy = max(maxy, pos[i][1])
        minx = min(minx, pos[i][0])
        miny = min(miny, pos[i][1])
    maxx = maxx + 0.1
    maxy = maxy + 0.1
    minx = minx - 0.1
    miny = miny - 0.1
    for i in range(len(nodes)):
        nodes[i]["x"] = (pos[i][0] - minx) / (maxx - minx)
        nodes[i]["y"] = (pos[i][1] - miny) / (maxy - miny)


"""
普通鱼眼放大
输入c是核心点index，int类型数据。graph是需要布局的图，AutoVis图数据
"""


def gfish(c, graph):  # pragma: no cover
    nodes = graph["nodes"]
    cx = nodes[c]["x"]  # 核心点坐标
    cy = nodes[c]["y"]
    for i in range(len(nodes)):
        if(i == c):
            continue
        nx = nodes[i]["x"]  # 点i坐标
        ny = nodes[i]["y"]
        # 计算边界点
        if(nx > cx):
            by = (ny - cy) * (1 - nx)/(nx - cx) + ny
            if(by < 0):
                by = 0
                bx = (nx - cx) * (0 - ny)/(ny - cy) + nx
            elif(by > 1):
                by = 1
                bx = (nx - cx) * (1 - ny)/(ny - cy) + nx
            else:
                bx = 1
        elif(nx < cx):
            by = (ny - cy) * (0 - nx)/(nx - cx) + ny
            if(by < 0):
                by = 0
                bx = (nx - cx) * (0 - ny)/(ny - cy) + nx
            elif(by > 1):
                by = 1
                bx = (nx - cx) * (1 - ny)/(ny - cy) + nx
            else:
                bx = 0
        elif(ny > cy):
            bx = cx
            by = 1
        else:
            bx = cx
            by = 0
        p = pow((pow(nx - cx, 2) + pow(ny - cy, 2)) /
                (pow(bx - cx, 2) + pow(by - cy, 2)), 0.5)  # 边nc/bc，长度比值
        m = 3
        p2 = (m + 1.) * p / (m * p + 1.)  # 非线性放大
        nodes[i]["x"] = cx + (bx - cx) * p2  # 鱼眼放大后点i坐标
        nodes[i]["y"] = cy + (by - cy) * p2


"""
共轭斜量法
Ax=b
求解x
"""


def cg(A, b, x):  # pragma: no cover
    r = b-np.dot(A, x)  # r=b-Ax         r也是是梯度方向
    p = np.copy(r)
    i = 0
    while(max(abs(r)) > 1.e-10 and i < 100):
        pap = np.dot(np.dot(A, p), p)
        if pap == 0:  # 分母太小时跳出循环
            return x
        alpha = np.dot(r, r)/pap  # 直接套用公式
        x1 = x + alpha*p
        r1 = r-alpha*np.dot(A, p)
        beta = np.dot(r1, r1)/np.dot(r, r)
        p1 = r1 + beta*p
        r = r1
        x = x1
        p = p1
        i = i+1
    return x


"""
求导w（xi-xj-s）2
对xi求导：wxi-wxj=ws
对xj求导：-wxi+wxj=-ws
"""


def addDiff(i, j, A, b, w, s):  # pragma: no cover
    A[i][i] += w
    A[i][j] -= w
    b[i] += s * w
    A[j][i] -= w
    A[j][j] += w
    b[j] -= s * w


"""
多核心鱼眼
输入c2是核心点index的列表，list<int>类型数据。graph是需要布局的图，AutoVis图数据。ws, wr, wt是三个参数，默认值为1。
"""


def mult_fish(c2, graph, ws=1, wr=1, wt=1):  # pragma: no cover
    focalArea = 0.2
    nodes = graph["nodes"]
    links = graph["links"]
    n = len(nodes)
    # 初始化导数方程
    Ax = np.zeros((n, n), dtype=float)
    bx = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    Ay = np.zeros((n, n), dtype=float)
    by = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)

    # 对每个核心节点计算损失值1
    for c in c2:
        g = copy.deepcopy(graph)
        gfish(c, g)  # 普通鱼眼
        newNodes = g["nodes"]

        for i in range(len(links)):
            snode = links[i][source_name]
            tnode = links[i][target_name]
            if(snode == tnode):
                continue
            oldw = pow(pow(nodes[snode]["x"] - nodes[tnode]["x"], 2) +
                       pow(nodes[snode]["y"] - nodes[tnode]["y"], 2), 0.5)
            s = [nodes[snode]["x"] - nodes[tnode]["x"],
                 nodes[snode]["y"] - nodes[tnode]["y"]]  # 初始布局边方向
            neww = pow(pow(newNodes[snode]["x"] - newNodes[tnode]["x"], 2) + pow(
                newNodes[snode]["y"] - newNodes[tnode]["y"], 2), 0.5)  # 普通鱼眼边长
            addDiff(snode, tnode, Ax, bx, ws, s[0]*neww/oldw)  # 导数方程中添加损失值1的导数
            addDiff(snode, tnode, Ay, by, ws, s[1]*neww/oldw)

    # 计算损失值3
    for i in range(len(nodes)):
        Ax[i][i] += 1. * wt
        bx[i] += nodes[i]["x"] * wt
        Ay[i][i] += 1. * wt
        by[i] += nodes[i]["y"] * wt

    for c in c2:
        # 确定核心区域
        focalNodes = []
        for i in range(len(nodes)):
            d = pow(pow(nodes[i]["x"] - nodes[c]["x"], 2) +
                    pow(nodes[i]["y"] - nodes[c]["y"], 2), 0.5)
            if(d <= focalArea):
                focalNodes.append(i)
        # 计算损失值2
        for i in range(len(focalNodes)):
            for j in range(i+1, len(focalNodes)):
                oldw = pow(pow(nodes[i]["x"] - nodes[j]["x"], 2) +
                           pow(nodes[i]["y"] - nodes[j]["y"], 2), 0.5)
                neww = 0.07  # 固定点间距
                if(oldw < neww):
                    s = [nodes[i]["x"] - nodes[j]["x"], nodes[i]
                         ["y"] - nodes[j]["y"]]  # 初始布局点方位
                    addDiff(i, j, Ax, bx, wr, s[0]*neww/oldw)
                    addDiff(i, j, Ay, by, wr, s[1]*neww/oldw)

    x = cg(Ax, bx, x)  # 求解导数方程，得点位置
    y = cg(Ay, by, y)
    for i in range(len(nodes)):
        nodes[i]["x"] = x[i]
        nodes[i]["y"] = y[i]


"""
struct_aware鱼眼
输入c是核心点index，list<int>类型数据。graph是需要布局的图，AutoVis图数据。ws, wr, wt是三个参数，默认值为1。
"""


def struct_aware_fish(c, graph, ws=1, wr=1, wt=1):  # pragma: no cover
    focalArea = 0.2
    nodes = graph["nodes"]
    links = graph["links"]
    n = len(nodes)
    # 初始化导数方程
    Ax = np.zeros((n, n), dtype=float)
    bx = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    Ay = np.zeros((n, n), dtype=float)
    by = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)

    g = copy.deepcopy(graph)
    gfish(c, g)  # 普通鱼眼
    newNodes = g["nodes"]

    # 计算损失值1
    for i in range(len(links)):
        snode = links[i][source_name]
        tnode = links[i][target_name]
        if(snode == tnode):
            continue
        oldw = pow(pow(nodes[snode]["x"] - nodes[tnode]["x"], 2) +
                   pow(nodes[snode]["y"] - nodes[tnode]["y"], 2), 0.5)
        s = [nodes[snode]["x"] - nodes[tnode]["x"],
             nodes[snode]["y"] - nodes[tnode]["y"]]  # 初始布局边方向
        neww = pow(pow(newNodes[snode]["x"] - newNodes[tnode]["x"], 2) +
                   pow(newNodes[snode]["y"] - newNodes[tnode]["y"], 2), 0.5)  # 普通鱼眼边长
        addDiff(snode, tnode, Ax, bx, ws, s[0]*neww/oldw)
        addDiff(snode, tnode, Ay, by, ws, s[1]*neww/oldw)

    # 计算损失值3
    for i in range(len(nodes)):
        Ax[i][i] += 1. * wt
        bx[i] += nodes[i]["x"] * wt
        Ay[i][i] += 1. * wt
        by[i] += nodes[i]["y"] * wt

    # 确定核心区域
    focalNodes = []
    for i in range(len(nodes)):
        d = pow(pow(nodes[i]["x"] - nodes[c]["x"], 2) +
                pow(nodes[i]["y"] - nodes[c]["y"], 2), 0.5)
        if(d <= focalArea):
            focalNodes.append(i)
    # 计算损失值2
    for i in range(len(focalNodes)):
        for j in range(i+1, len(focalNodes)):
            oldw = pow(pow(nodes[i]["x"] - nodes[j]["x"], 2) +
                       pow(nodes[i]["y"] - nodes[j]["y"], 2), 0.5)
            neww = 0.07  # 固定点间距
            if(oldw < neww):
                s = [nodes[i]["x"] - nodes[j]["x"], nodes[i]
                     ["y"] - nodes[j]["y"]]  # 初始布局点方位
                addDiff(i, j, Ax, bx, wr, s[0]*neww/oldw)
                addDiff(i, j, Ay, by, wr, s[1]*neww/oldw)

    x = cg(Ax, bx, x)  # 求解导数方程，得点位置
    y = cg(Ay, by, y)
    for i in range(len(nodes)):
        nodes[i]["x"] = x[i]
        nodes[i]["y"] = y[i]


def mult_layout_without_fish(c, graph):  # pragma: no cover
    g = copy.deepcopy(graph)
    v = g["nodes"]
    e = g["links"]
    n = len(v)

    v.append({'type': ""})  # 添加虚拟核心点
    for i in c:
        e.append({source_name: i, target_name: n})  # 添加虚拟核心点与核心点的连线

    getStartLayout(n, g, 2)  # 计算初始布局
    for i in range(n):
        graph["nodes"][i] = g["nodes"][i]  # 保留除虚拟核心点外的布局结果
    return graph


"""
整体布局算法
输入c是核心点index的列表，list<int>类型数据。graph是需要布局的图，AutoVis图数据。
"""


def mult_layout(c, graph):  # pragma: no cover
    g = copy.deepcopy(graph)
    v = g["nodes"]
    e = g["links"]
    n = len(v)

    v.append({'type': ""})  # 添加虚拟核心点
    for i in c:
        e.append({source_name: i, target_name: n})  # 添加虚拟核心点与核心点的连线

    getStartLayout(n, g, 2)  # 计算初始布局
    for i in range(n):
        graph["nodes"][i] = g["nodes"][i]  # 保留除虚拟核心点外的布局结果
    mult_fish(c, graph)  # 鱼眼放大
    return graph


def repair_graph(graph):  # pragma: no cover
    if "nodes" not in graph:
        return None
    v = graph["nodes"]
    e = graph["links"]
    n = len(v)
    if n == 0:
        return None
    name_index = {}
    for i in range(n):
        if "index" not in v[i]:
            return None
        if "weight" not in v[i]:
            v[i]["weight"] = 0
        name_index[v[i]["index"]] = i  # 令点的index值与点在列表中的位置对应

    error_e = []
    for i in range(len(e)):
        if e[i][source_name] not in name_index or e[i][target_name] not in name_index:
            error_e.append(i)
        else:
            # 边中用index值确定点，替换成点在列表中的位置，方便取值
            e[i][source_name] = name_index[e[i][source_name]]
            e[i][target_name] = name_index[e[i][target_name]]
    for i in range(len(error_e)):
        del e[error_e[i] - i]
    return graph


def add_subgraph_center(graph, node_num, c):  # pragma: no cover
    # 将AutoVis图转化为networkx图
    nxG = nx.Graph()
    nodes = graph["nodes"]
    nxG.add_nodes_from(range(len(nodes)))
    links = graph["links"]
    for i in range(len(links)):
        if(links[i][source_name] != links[i][target_name]):
            nxG.add_edge(links[i][source_name], links[i][target_name])

    for p in nx.connected_components(nxG):
        p = list(p)
        c_nodes = p[np.argmax([node_num[i] for i in p])]
        if c_nodes not in c:
            c.append(c_nodes)
    return c


def simple_mult_layout(graph):  # pragma: no cover

    g = copy.deepcopy(graph)
    g = repair_graph(g)
    if g == None:
        return None

    c = []  # 核心点
    for i in range(len(g["nodes"])):
        if g["nodes"][i]["weight"] == 1:
            c.append(i)  # weight=1的点为核心点

    n = len(graph["nodes"])
    node_num = np.zeros(n)
    for i in range(len(g["links"])):
        node_num[g["links"][i][source_name]] += 1  # 计算每个点的出入度之和
        node_num[g["links"][i][target_name]] += 1

    c = add_subgraph_center(g, node_num, c)

    g = mult_layout(c, g)  # 调用整体布局算法

    for i in range(n):
        graph["nodes"][i] = g["nodes"][i]  # 将整体布局的点的位置写入graph中

    node_max = max(1, np.max(node_num))
    for i in range(n):
        graph["nodes"][i]["size"] = node_num[i]*7100. / \
            node_max + 2900  # 根据出入度大小，为点赋予size值，取值[100,2000]
    return graph
