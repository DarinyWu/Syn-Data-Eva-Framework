import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from collections import defaultdict
from utils import *

def mutual_info(x, y):
    return mutual_info_score(x, y)

def dp_mutual_info(x, y, epsilon, sensitivity=1.0):
    mi = mutual_info(x, y)
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon)
    return mi + noise

def build_dp_bn(data, epsilon):
    attrs = list(data.columns)
    tree = {}
    used = set()

    root = attrs[0]
    tree[root] = None
    used.add(root)

    while len(used) < len(attrs):
        best_pair = None
        best_score = -np.inf
        for x in used:
            for y in attrs:
                if y in used:
                    continue
                score = dp_mutual_info(data[x], data[y], epsilon / len(attrs))
                if score > best_score:
                    best_score = score
                    best_pair = (y, x)
        y, x = best_pair
        tree[y] = x
        used.add(y)
    return tree

def estimate_cpt(data, bn):
    cpts = {}
    for child, parent in bn.items():
        if parent is None:
            probs = data[child].value_counts(normalize=True).to_dict()
        else:
            probs = (
                data.groupby([parent, child]).size() /
                data.groupby(parent).size()
            ).fillna(0).to_dict()
        cpts[child] = probs
    return cpts

def sample_from_bn(bn, cpts, n):
    synthetic = []
    for _ in range(n):
        row = {}
        for attr in bn:
            parent = bn[attr]
            if parent is None:
                dist = cpts[attr]
                val = np.random.choice(list(dist.keys()), p=list(dist.values()))
            else:
                parent_val = row[parent]
                cond_dist = {k[1]: v for k, v in cpts[attr].items() if k[0] == parent_val}
                vals, probs = zip(*cond_dist.items()) if cond_dist else ([0], [1.0])
                probs = np.array(probs)
                probs = probs / probs.sum()
                val = np.random.choice(vals, p=probs)
            row[attr] = val
        synthetic.append(row)
    return pd.DataFrame(synthetic)

#Customize your Usage