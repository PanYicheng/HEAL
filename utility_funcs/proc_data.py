import pickle
import os

import numpy as np
import pandas as pd
import networkx as nx
from scipy import interpolate


def safe_dump_obj(obj, fname):
    """Dump the object to pickle file fname. Create parent directory if needed.

    Args:
        obj (Anything): the python object to dump.
        fname (str): the pickle file's full path name.
    """
    if fname is None or obj is None:
        return
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def aggregate(a, n=3):
    cumsum = np.cumsum(a, dtype=float)
    ret = []
    for i in range(-1, len(a) - n, n):
        low_index = i if i >= 0 else 0
        ret.append(cumsum[low_index + n] - cumsum[low_index])
    return ret


tcdf_data_path = "/workspace/TCDF/data/"
timeseries_files = [
    "manyinputs_returns30007000_header.csv",
    "random-rels_20_1A_returns30007000_header.csv",
    "random-rels_20_1B_returns30007000_header.csv",
    "random-rels_20_1C_returns30007000_header.csv",
    "random-rels_20_1D_returns30007000_header.csv",
    "random-rels_20_1E_returns30007000_header.csv",
    "random-rels_40_1_returns30007000_header.csv",
    "random-rels_40_1_3_returns30007000_header.csv",
    "random-rels_20_1_3_returns30007000_header.csv",
]
gt_files = [
    "manyinputs.csv",
    "random-rels_20_1A.csv",
    "random-rels_20_1B.csv",
    "random-rels_20_1C.csv",
    "random-rels_20_1D.csv",
    "random-rels_20_1E.csv",
    "random-rels_20_1_3.csv",
    "random-rels_40_1.csv",
    "random-rels_40_1_3.csv",
]

data_store = {"finance": (timeseries_files, gt_files)}


def load_tcdf_data(dir="finance"):
    """Load the TCDF paper's sythetic time series data with ground truth causal graph.

    Params:
        dir: which datasets to load. Can be "finance" or "fmri". See more in the paper.
    Returns:
        datasets: A list of DataFrames containing the datasets. Each DataFrame's shape is [T, N],
            where T is the number of data samples and N is the number of variables.
        gt_graphs: A list of ground truth causal graphs. Edge also has a 'lag' attribute indicating
            the causal delays.
    """
    datasets = []
    gt_graphs = []
    if dir in data_store:
        for ts_f, gt_f in zip(*data_store[dir]):
            data = pd.read_csv(os.path.join(tcdf_data_path, "Finance", ts_f), header=0)
            N = len(data.columns)
            gt_graph = nx.DiGraph()
            gt_graph.add_nodes_from(range(N))
            with open(os.path.join(tcdf_data_path, "Finance", gt_f), "rt") as f:
                for line in f:
                    s, e, lag = line.strip().split(",")
                    gt_graph.add_edge(int(s), int(e), lag=lag)
            datasets.append(data)
            gt_graphs.append(gt_graph)
    elif dir == "fmri":
        for i in range(1, 29, 1):
            data = pd.read_csv(
                os.path.join(tcdf_data_path, "fMRI", f"timeseries{i}.csv"), header=0
            )
            N = len(data.columns)
            gt_graph = nx.DiGraph()
            gt_graph.add_nodes_from(range(N))
            with open(
                os.path.join(tcdf_data_path, "fMRI", f"sim{i}_gt_processed.csv"), "rt"
            ) as f:
                for line in f:
                    s, e, lag = line.strip().split(",")
                    gt_graph.add_edge(int(s), int(e), lag=lag)
            datasets.append(data)
            gt_graphs.append(gt_graph)
    else:
        print("No such datasets.")
        exit(1)
    return datasets, gt_graphs


def load_pairs_data():
    """Load the pairs dataset from https://webdav.tuebingen.mpg.de/cause-effect/.
    Returns:
        dfs: A list of DataFrames containing the datasets. Each DataFrame's shape is [T, N],
            where T is the number of data samples and N is the number of variables.
        relations: A list of tuples (cause, effect). cause or effect could be 'x' or 'y'.
            'x' is the first column and 'y' is the second column.
    """
    data_root = 'data/pairs'

    idx = 1
    dfs = []
    relations = []
    for idx in range(1, 109):
        # Problem data.
        if idx in [52, 54, 55, 71, 72, 81, 82, 83, 86, 105]:
            continue
        data_fname = os.path.join(data_root, "pair{:04}.txt".format(idx))
        desc_fname = os.path.join(data_root, "pair{:04}_des.txt".format(idx))
        # print(data_fname)
        a = np.genfromtxt(data_fname)
        df = pd.DataFrame(a)
        with open(desc_fname, "rt") as f:
            lines = f.readlines()
        lines = [line.lower() for line in lines]
    
        def findline(s):
            for line in lines:
                for i in s:
                    if line.find(i) != -1:
                        ret = line.split(i)[1].strip()
                        return ret
            return None
        x_line = findline(["x:", "x =", "(x):"])
        y_line = findline(["y:", "y =", "(y):"])
        colnames = []
        
        if x_line is not None and '\t' in x_line:
            colnames.extend(x_line.split('\t'))
        elif x_line is not None:
            colnames.append(x_line)
        if y_line is not None and '\t' in y_line:
            colnames.extend(y_line.split('\t'))
        elif y_line is not None:
            colnames.append(y_line)
        if df.shape[1] == len(colnames):
            df.columns = colnames
            # print(df.columns)
        elif df.shape[1] > len(colnames):
            print("Not enough column names in data description!")
            break
        else:
            print("Too many column names in data description!")
            break
        
        gt_str = None
        if 'ground truth:\n' in lines:
            gt_str = lines[lines.index('ground truth:\n')+1].strip()
            # print(gt_str)
        else:
            for line in lines:
                if '-->' in line:
                    gt_str = line.strip()
                    # print(gt_str)
                    break
                elif 'ground truth' in line:
                    gt_str = line.lstrip('ground truth').strip(': \n')
                    # print(gt_str)
                    break
        if gt_str is None:
            print("No groundtruth found!")
            break
        if '-->' in gt_str:
            l, r = gt_str.split('-->')
        elif '->' in gt_str:
            l, r = gt_str.split('->')
        l = l.strip()
        r = r.strip()
        
        relation = (l, r)
        dfs.append(df)
        relations.append(relation)
    return dfs, relations
