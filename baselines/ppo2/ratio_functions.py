import numpy as np

TS = [1, 0.5, 0.2, 0.1, 1, 0.5]
SCALES = [2, 1, 1, 0.5, 1]

def compute_ratio_graph(v):
    v1, v2, v3, v4 = v[0], v[1], v[2], v[3]
    p11 = (1 - v1)
    p22 = (1 - v2)
    p33 = 1 - v3
    p44 = 1 - v4

    p01 = TS[0]
    p02 = TS[1]
    p03 = TS[2]
    p04 = TS[3]

    p12 = (1 - p11)*TS[4]
    p23 = (1 - p22)*TS[4]

    p1 = p01*p11
    p2 = p02*p22 + p01*p12*p22 
    p3 = p03*p33 + p01*p12*p23*p33
    p4 = p04*p44 
    p5 = p4*0.1

    return [p1, p2, p3, p4, p5]

def compute_ratio_auto(v):
    v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4]
    p11 = (1 - v1)
    p22 = (1 - v2)
    p33 = 1 - v3
    p44 = 1 - v4
    p55 = 1 - v5

    p1 = p11
    p2 = p22
    p3 = p33
    p4 = p44
    p5 = p55

    return [p1, p2, p3, p4, p5]