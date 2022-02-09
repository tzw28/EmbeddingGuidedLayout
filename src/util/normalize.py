import copy


def normalize_list(l, y1, y2):
    if len(l) == 0:
        return []
    lc = copy.deepcopy(l)
    x1 = lc[0]
    x2 = lc[0]
    for x in lc:
        if x < x1:
            x1 = x
        if x > x2:
            x2 = x
    for i, x in enumerate(lc):
        if x1 == x2:
            y = (y1 + y2) / 2
        else:
            y = (x-x1) / (x2-x1) * (y2-y1) + y1
        lc[i] = y
    return lc
