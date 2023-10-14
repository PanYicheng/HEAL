import numpy as np
import pickle
from dycause_lib.causal_graph_build import get_segment_split, get_overlay_count
from collections import defaultdict
from dycause_lib.causal_graph_build import normalize_by_column, normalize_by_row
from tigramite.independence_tests import ParCorr, GPDC, CMIknn
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import time
from dycause_lib.rca import case_rca_backtrace
import utility_funcs.graph_draw as gd
from utility_funcs.evaluation_function import print_prk_acc


def build_intervals_oneside(local_results, significant_thres, local_length, step, N, verbose=False):
    list_segment_split = get_segment_split(local_length, step)
    for x_i in range(N):
        for y_i in range(N):
            if x_i == y_i:
                continue
            key = "{}->{}".format(x_i, y_i)
#             print(key, local_results[key].keys())
            if key not in local_results:
                continue
            array_results_YX, array_results_XY = local_results[
                key]['result_YX'], local_results[key]['result_XY']
            array_results_YX = np.abs(array_results_YX)
            array_results_XY = np.abs(array_results_XY)
            nrows, ncols = array_results_YX.shape
            intervals = []
            pvalues = []
            for i in range(nrows):
                for j in range(i + 1, ncols):
                    if (abs(array_results_YX[i, j]) < significant_thres):
                        intervals.append(
                            (list_segment_split[i], list_segment_split[j]))
                        pvalues.append(
                            (array_results_YX[i, j], array_results_XY[i, j]))
#             if verbose:
#                 print(key, intervals)
            ordered_intervals = list(zip(pvalues, intervals))
            ordered_intervals.sort(key=lambda x: (x[0][0], -x[0][1]))
            local_results[key]['intervals_oneside'] = ordered_intervals


def build_intervals_dycause_level(local_results, significant_thres, local_length, step, N, level=None, verbose=False):
    list_segment_split = get_segment_split(local_length, step)
    for x_i in range(N):
        for y_i in range(N):
            if x_i == y_i:
                continue
            key = "{}->{}".format(x_i, y_i)
#             print(key, local_results[key].keys())
            if key not in local_results:
                continue
            array_results_YX, array_results_XY = local_results[
                key]['result_YX'], local_results[key]['result_XY']
            array_results_YX = np.abs(array_results_YX)
            array_results_XY = np.abs(array_results_XY)
            nrows, ncols = array_results_YX.shape
            intervals = []
            pvalues = []
            for i in range(nrows):
                for j in range(i + 1, ncols):
                    if level is not None and j-i > level:
                        continue
                    if (abs(array_results_YX[i, j]) < significant_thres) and (
                        array_results_XY[i, j] >= significant_thres
                        # p < 0 means that the actual value is not computed. So we also assume the causality does not exist.
                        or array_results_XY[i, j] < 0
                    ):
                        intervals.append(
                            (list_segment_split[i], list_segment_split[j]))
                        pvalues.append(
                            (array_results_YX[i, j], array_results_XY[i, j]))
#             if verbose:
#                 print(key, intervals)
            ordered_intervals = list(zip(pvalues, intervals))
            ordered_intervals.sort(key=lambda x: (x[0][0], x[0][1]))
            local_results[key]['intervals_dycause_level'] = ordered_intervals


def build_intervals_special(local_results, significant_thres, local_length, step, N, level=None, verbose=False):
    list_segment_split = get_segment_split(local_length, step)
    for x_i in range(N):
        for y_i in range(N):
            if x_i == y_i:
                continue
            key = "{}->{}".format(x_i, y_i)
#             print(key, local_results[key].keys())
            if key not in local_results:
                continue
            array_results_YX, array_results_XY = local_results[
                key]['result_YX'], local_results[key]['result_XY']
            array_results_YX = np.abs(array_results_YX)
            array_results_XY = np.abs(array_results_XY)
            nrows, ncols = array_results_YX.shape
            intervals = []
            pvalues = []
            for i in range(nrows):
                for j in range(i + 1, ncols):
                    if level is not None and j-i > level:
                        continue
                    if (abs(array_results_YX[i, j]) < significant_thres and abs(array_results_XY[i, j]) < significant_thres):
                        intervals.append(
                            (list_segment_split[i], list_segment_split[j]))
                        pvalues.append(
                            (array_results_YX[i, j], array_results_XY[i, j]))
#             if verbose:
#                 print(key, intervals)
            ordered_intervals = list(zip(pvalues, intervals))
            ordered_intervals.sort(key=lambda x: (x[0][0], x[0][1]))
            local_results[key]['intervals_special'] = ordered_intervals


def generate_DCC(local_results, T, N, interval_key="intervals", agg_func=None,
                 verbose=False):
    """Generate dynamic causality curve (DCC) between two services by overlaying 
    intervals.

    Args:
        local_results: Causal discovery temp results.
        T: Data length in time dimension.
        N: Num of time series.
        interval_key: the intervals key in the local_results dict.
        agg_func: The aggregation function to merge intervals into dynamic 
            causality curves. Default is the sum aggregation function.
        verbose: whether print intermidiate information.
    """
    DCC = defaultdict(int)
    if agg_func is None:
        agg_func = get_overlay_count
    for x_i in range(N):
        for y_i in range(N):
            if y_i == x_i:
                continue
            key = f"{x_i}->{y_i}"
            if key not in local_results:
                continue
            intervals = local_results[key][interval_key]
#             from IPython.core.debugger import set_trace
#             set_trace() # 断点位置
            try:
                if len(intervals) != 0 and not isinstance(intervals[0][0], int):
                    #                 print("not isinstance(intervals[0], int)", type(intervals[0]))
                    intervals = [_[1] for _ in intervals]
            except Exception:
                from IPython.core.debugger import set_trace
                set_trace()  # 断点位置
            if verbose:
                print(key, intervals)
            overlay_counts = agg_func(T, intervals)
            DCC[key] = overlay_counts
    return DCC


def get_dir_hints_from_dcc(dcc, N, thr):
    """Hints from asymmetric patterns of dcc
    """
    hints = []
    for i in range(N):
        for j in range(N):
            v0 = float(np.sum(dcc[f"{i}->{j}"]))
            v1 = float(np.sum(dcc[f"{j}->{i}"]))
            if v0 - v1 >= thr:
                hints.append((i, j, v0-v1))
            elif v1 - v0 >= thr:
                hints.append((j, i, v1-v0))
    hints = sorted(hints)
    if len(hints) == 0:
        print("No hints generated! Please adjust parameters or check errors.")
        return []
    i = 0
    r = [hints[0]]
    for i in range(len(hints)-1):
        if hints[i+1] == r[-1]:
            continue
        else:
            r.append(hints[i+1])
    return r


def remove_double_direction_hints(hints):
    _d = {}
    for i, j, v in hints:
        _d[i, j] = 1
    removing = []
    for k in _d:
        if (k[1], k[0]) in _d:
            print(f"Double directed hints: {k[0]}, {k[1]}")
            removing.append((k[0], k[1]))
            removing.append((k[1], k[0]))
    r = []
    for k in _d:
        if k not in removing:
            r.append((k[0], k[1], 0))
    return r


def global_thresholding(DCC, N, normal_axis='col', ntop=100, only_source_ids=None,
                        only_target_ids=None, hints=None, return_candidates=False,
                        verbose=False):
    '''Construct a graph according to the dynamic causality curves. The graph 
    will follow the direction hints provided in only_source_ids, only_target_ids
    and hints. The algorithm logic is first sorting all edges by the dcc strength
    and then selecting the top ntop ones.
    Params:
        DCC:
        N:
        normal_axis:
        ntop:
        only_source_ids: the graph nodes that can only be source.
        only_target_ids: the graph nodes that can only be source.
        hints:
    '''
    edge = []
    edge_weight = {}
    vals = []
    for x_i in range(N):
        for y_i in range(N):
            if x_i != y_i:
                key = "{0}->{1}".format(x_i, y_i)
                arr = DCC[key]
                vals.append((x_i, y_i, float(np.sum(arr))))
    vals.sort(key=lambda x: x[2], reverse=True)
    if verbose:
        print("Total num of edges:", len([0 for _ in vals if _[2] > 0]))
    # Build a hint lookup dict
    hints_d = {}
    if hints is not None:
        for x, y, v in hints:
            hints_d[(x, y)] = v
    i = 0

    def check_edge(x, y):
        if only_source_ids is not None and y in only_source_ids:
            return False
        elif only_target_ids is not None and x in only_target_ids:
            return False
        elif (y, x) in hints_d:
            return False
        return True
    while len(edge) < ntop and i < len(vals):
        x, y, v = vals[i]
        if check_edge(x, y):
            #             print(f"Legal edge: {x:2} -> {y:2} {v}")
            edge.append((x, y))
            edge_weight[(x, y)] = v
#         else:
#             print(f"Illegal edge: {x:2} -> {y:2} {v}")
        i += 1
    candidate_edges = []
    while i < len(vals):
        x, y, v = vals[i]
        if check_edge(x, y):
            candidate_edges.append(vals[i])
        i += 1
#     print(edge)
    # Make the transition matrix with edge weight estimation
    transition_matrix = np.zeros([N, N])
    for key, val in edge_weight.items():
        x, y = key
        transition_matrix[x, y] = val
    if normal_axis == 'col':
        transition_matrix = normalize_by_column(transition_matrix)
    elif normal_axis == 'row':
        transition_matrix = normalize_by_row(transition_matrix)
    elif normal_axis == 'none':
        pass
    else:
        raise NotImplementedError(
            "No such normal_axis. Can only be col, row, or none.")
    if return_candidates:
        return transition_matrix, candidate_edges
    return transition_matrix


def calculate_parcorr(mat, df):
    N = mat.shape[0]
    all_nodes = set(range(N))
    parcorr_res = {}
    p = ParCorr()
    for i, j in zip(*mat.nonzero()):
        conds = list(range(N))
        conds.remove(i)
        conds.remove(j)
        cols = [i, j] + conds
        arr = df.iloc[:, cols].to_numpy().T
        try:
            pvalue = p.get_dependence_measure(arr, [])
            parcorr_res[i, j] = pvalue
        except ValueError as e:
            parcorr_res[i, j] = None
    return parcorr_res


def calculate_GPDC(mat, graph_idx_to_data_idx, df):
    N = mat.shape[0]
    all_nodes = set(range(N))
    parcorr_res = {}
    p = GPDC()
    for i, j in zip(*mat.nonzero()):
        conds = list(range(N))
        conds.remove(i)
        conds.remove(j)
        cols = [i, j] + conds
        cols = [graph_idx_to_data_idx[_] for _ in cols]
        arr = df.iloc[:, cols].to_numpy().T
        pvalue = p.get_dependence_measure(arr, [])
        parcorr_res[i, j] = pvalue
    return parcorr_res


def calculate_CMIknn(mat, graph_idx_to_data_idx, df):
    N = mat.shape[0]
    all_nodes = set(range(N))
    parcorr_res = {}
    p = CMIknn()
    for i, j in zip(*mat.nonzero()):
        conds = get_cond_vars(mat, i, j)
        cols = [i, j] + conds
        xyz = np.array([0, 1] + [2] * len(conds))
        cols = [graph_idx_to_data_idx[_] for _ in cols]
        arr = df.iloc[:, cols].to_numpy().T
        pvalue = p.get_dependence_measure(arr, xyz)
        parcorr_res[i, j] = pvalue
    return parcorr_res


def get_cond_vars(mat, x, y):
    r = set()
    # Nodes x --> ?
    for i in mat[x, :].nonzero()[0]:
        r.add(i)
    # Nodes y --> ?
    for i in mat[y, :].nonzero()[0]:
        r.add(i)
    # Nodes ? --> x
    for i in mat[:, x].nonzero()[0]:
        r.add(i)
    # Nodes ? --> y
    for i in mat[:, y].nonzero()[0]:
        r.add(i)
    try:
        r.remove(x)
    except KeyError:
        pass
    try:
        r.remove(y)
    except KeyError:
        pass
    return list(r)


def calculate_CMIknn_shuffled(mat, df):
    N = mat.shape[0]
    corr_res_cond = {}
    corr_res_cond_shuffled = {}
    analyze_pairs = list(zip(*mat.nonzero()))
    for i, j in tqdm(analyze_pairs, total=len(analyze_pairs)):
        p = CMIknn(n_jobs=20)
        conds = get_cond_vars(mat, i, j)
        cols = [i, j] + conds
        xyz = np.array([0, 1] + [2] * len(conds))
        arr = df.iloc[:, cols].to_numpy().T
        pvalue = p.get_dependence_measure(arr, xyz)
        corr_res_cond[i, j] = pvalue
        # Conditional CMIknn shuffled
        pvalue = p.get_shuffle_significance(arr, xyz, corr_res_cond[i, j])
        corr_res_cond_shuffled[i, j] = pvalue
    return corr_res_cond_shuffled, corr_res_cond


def CMIknn_subprocess(arr, xyz, i, j, result_dict):
    p = CMIknn(n_jobs=20)
    pvalue1 = p.get_dependence_measure(arr, xyz)
    pvalue2 = p.get_shuffle_significance(arr, xyz, pvalue1)
    result_dict[i, j] = (pvalue1, pvalue2)
    return pvalue1, pvalue2


def calculate_CMIknn_shuffled_mp(mat, df, max_workers=15, verbose=False):
    N = mat.shape[0]
    corr_res_cond = {}
    corr_res_cond_shuffled = {}

    analyze_pairs = list(zip(*mat.nonzero()))
    executor = ProcessPoolExecutor(max_workers=max_workers)
    manager = Manager()
    result_dict = manager.dict()
    futures = []
    if verbose:
        pbar = tqdm(total=len(analyze_pairs))
    for i, j in analyze_pairs:
        #         conds = list(range(N))
        #         try:
        #             conds.remove(i)
        #             conds.remove(j)
        #         except ValueError as e:
        #             pass
        conds = get_cond_vars(mat, i, j)
        cols = [i, j] + conds
        xyz = np.array([0, 1] + [2] * len(conds))
        arr = df.iloc[:, cols].to_numpy().T
        futures.append((i, j,
                        executor.submit(CMIknn_subprocess, arr, xyz, i, j, result_dict)))
    if verbose:
        for fut in as_completed([_[2] for _ in futures]):
            pbar.update(1)
        pbar.close()
    executor.shutdown(wait=True)
    for i, j, fut in futures:
        try:
            res = fut.result()
            corr_res_cond[i, j] = res[0]
            # Conditional CMIknn shuffled
            corr_res_cond_shuffled[i, j] = res[1]
        except Exception:
            print('CMI between {} and {} failed!'.format(i, j))
    return corr_res_cond_shuffled, corr_res_cond


def cond_test_filtering(mat, candidates, corr_res, p_thr, comp_func):
    mat_copy = mat.copy()
    candidates_ind = 0
    for e, p in corr_res.items():
        if comp_func(abs(p), p_thr):
            mat_copy[e] = 0
            new_i, new_j, new_w = candidates[candidates_ind]
            mat_copy[new_i, new_j] = new_w
            candidates_ind += 1
    return mat_copy


def cond_filter_search(mat, candidates, corr_res, verbose=False, comp_func=None):
    if comp_func is None:
        def comp_func(x, y): return x > y
    search_exps = []
    for p_thr in np.arange(0.1, 1.0, 0.1):
        if verbose:
            print('{:-^60}'.format(f' p ? {p_thr:.1f} '))
        tic = time.time()
        filtered_mat = cond_test_filtering(
            mat, candidates, corr_res, p_thr, comp_func)
        toc = time.time() - tic
        search_exps.append({
            'p_thr': p_thr,
            'filtered_mat': filtered_mat,
            'time_info': {'Cond_Filter': toc}
        })
    return search_exps


def our_rca_method(dycause_pkl_name, input_df, all_columns, only_source_ids,
                   only_target_ids, ntop,
                   entry, root_causes,
                   mean_method, topk_path, prob_thres, num_sel_node,
                   use_cond, p_thr, comp_func,
                   direction_hint_thres=250,
                   normal_profile_hints_filtered=None, verbose=True):
    with open(dycause_pkl_name, "rb") as f:
        exp_ret = pickle.load(f)

    time_info = {'DyCause': exp_ret["time_stat_dict"]
                 ["Construct-Impact-Graph-Phase"]}

    sign = exp_ret['sign']
    step = exp_ret['step']

    build_intervals_special(
        exp_ret['local_results'], sign, input_df.shape[0], step, input_df.shape[1])
    exp_ret['dcc_special'] = generate_DCC(
        exp_ret['local_results'], input_df.shape[0], input_df.shape[1], interval_key="intervals_special")

    tic = time.time()
    data_idx_to_graph_idx, graph_idx_to_data_idx = gd.construct_d2g_map_dict(list(input_df.columns),
                                                                             all_columns)
    exp_ret["filtered_dcc"] = gd.filter_dcc(
        exp_ret["dcc"], len(all_columns), graph_idx_to_data_idx)
    exp_ret["filtered_dcc_special"] = gd.filter_dcc(
        exp_ret["dcc_special"], len(all_columns), graph_idx_to_data_idx)
    filtered_df = gd.filter_df(
        input_df, graph_idx_to_data_idx, len(all_columns))
    # hints from anomaly dcc
    hints = get_dir_hints_from_dcc(
        exp_ret["filtered_dcc"], len(all_columns), direction_hint_thres)
    if normal_profile_hints_filtered is not None:
        # merge hints from normal profile
        hints = hints + normal_profile_hints_filtered
        hints = remove_double_direction_hints(hints)

    mat, candidates = global_thresholding(exp_ret["filtered_dcc_special"], len(all_columns), normal_axis='none',
                                          ntop=ntop, only_source_ids=only_source_ids,
                                          only_target_ids=only_target_ids,
                                          hints=hints,
                                          return_candidates=True
                                          )
    toc = time.time() - tic
    time_info['Graph'] = toc
    gd.mat_summary(mat)

    tic = time.time()
    if use_cond == 'cmiknn120':
        cond_corr_res, cond_cmi_res = calculate_CMIknn_shuffled_mp(
            mat, filtered_df.iloc[-120:, :], max_workers=10, verbose=True)
    elif use_cond == 'cmiknn':
        cond_corr_res, cond_cmi_res = calculate_CMIknn_shuffled_mp(
            mat, filtered_df, max_workers=10, verbose=True)
    elif use_cond == 'parcorr':
        cond_corr_res = calculate_parcorr(mat, filtered_df)
    else:
        time_info['Cond'] = 0
    if use_cond in ['cmiknn120', 'cmiknn', 'parcorr']:
        time_info['prev_mat'] = mat
        mat = cond_test_filtering(
            mat, candidates, cond_corr_res, p_thr, comp_func)
        toc = time.time() - tic
        time_info['Cond'] = toc

    tic = time.time()
    ranked_nodes, ranked_paths, prks, acc = case_rca_backtrace(
        mat, entry, root_causes, filtered_df,
        mean_method=mean_method, topk_path=topk_path, prob_thres=prob_thres, num_sel_node=num_sel_node)
    toc = time.time() - tic
    time_info['RCA'] = toc
    time_info['ranked_nodes'] = ranked_nodes
    time_info['ranked_paths'] = ranked_paths
    time_info['mat'] = mat
    if verbose:
        print_prk_acc(prks, acc)
        total_t = sum(
            [time_info[k] for k in ['DyCause', 'Graph', 'Cond', 'RCA'] if k in time_info])
        print('Total: {:.4f}'.format(total_t),
              '({})'.format(', '.join([f'{k}: {time_info[k]:.4f}' for k in ['DyCause', 'Graph', 'Cond', 'RCA'] if k in time_info])))
    return prks, acc, time_info


def backtrace_param_search(mat, entry, root_causes, filtered_df):
    ret_exps = []
    for mean_method in ["arithmetic", "geometric", "harmonic"]:
        for topk_path in [200, 300, 400]:
            for prob_thres in [0.2, 0.4, 0.6]:
            # for prob_thres in [0.0]:
                for num_sel_node in [1, 2, 3, 4, 5]:
                    params = {
                        'mean_method': mean_method,
                        'topk_path': topk_path,
                        'prob_thres': prob_thres,
                        'num_sel_node': num_sel_node,
                        'out_of_path_nodes': True
                    }
                    tic = time.time()
                    ranked_nodes, ranked_paths, prks, acc = case_rca_backtrace(
                        mat, entry, root_causes, filtered_df, **params)
                    toc = time.time() - tic
                    params.update({
                        # 'ranked_nodes': ranked_nodes,
                        # 'ranked_paths': ranked_paths,
                        'prks': prks,
                        'acc': acc,
                        'time_info': {
                            'RCA': toc
                        }
                    })
                    ret_exps.append(params)
    return ret_exps


def search_rca_backtrace_params(mat, candidates, corr_res_cond_shuffled,
                                entry, root_causes, filtered_df,
                                time_info,
                                verbose=True, comp_func=lambda x, p: x > p):

    # Search backtrae parameters with default graph
    ret_exps = backtrace_param_search(mat, entry, root_causes, filtered_df)
    case_rca_exp_res = [d for d in ret_exps]
    for d in case_rca_exp_res:
        for k in ['DyCause', 'Graph']:
            d['time_info'][k] = time_info[k]

    if corr_res_cond_shuffled is not None:
        # Search backtrace parameters with CMIknn filtered graphs
        search_exps = cond_filter_search(
            mat, candidates, corr_res_cond_shuffled, verbose=verbose, comp_func=comp_func)
        if verbose:
            pbar = tqdm(total=len(search_exps))
        for cond_d in search_exps:
            _ret_exps = backtrace_param_search(
                cond_d['filtered_mat'], entry, root_causes, filtered_df)

            for _d in _ret_exps:
                for k in ['p_thr', 'filtered_mat']:
                    _d[k] = cond_d[k]
                _d['time_info']['Cond_Filter'] = cond_d['time_info']['Cond_Filter']
                for k in ['DyCause', 'Graph', 'Cond']:
                    _d['time_info'][k] = time_info[k]
            case_rca_exp_res.extend(_ret_exps)
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
    return case_rca_exp_res
