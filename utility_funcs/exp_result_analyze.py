import numpy as np
from tabulate import tabulate
from utility_funcs.evaluation_function import print_prk_acc


def print_case_avg_max_perf(case_result_dict, verbose=True):
    prks_dict = {}
    acc_dict = {}
    time_arr = []
    for k in case_result_dict:
        prks_dict[f'{k}'] = []
        acc_dict[f'{k}'] = []
        time_arr.append([])
        for d in case_result_dict[k]:
            if d['prks'] is None:
                prks_dict[f'{k}'].append([0.0 for _ in range(10)])
                acc_dict[f'{k}'].append(0.0)
                time_arr[-1].append(0.0)
            else:
                prks_dict[f'{k}'].append(d['prks'])
                acc_dict[f'{k}'].append(d['acc'])
                time_arr[-1].append(sum(d['time_info'].values()))
    acc_counts = [len(acc_dict[k]) for k in acc_dict]
    assert np.unique(acc_counts).shape[0] == 1, "Not all cases have the same number of exps!"
    prks_arr = np.array(list(prks_dict.values()))
    acc_arr = np.array(list(acc_dict.values()))
    prks_avg_arr = np.mean(prks_arr, axis=2)
    acc_prkavg_arr = np.stack([acc_arr, prks_avg_arr], axis=2)
    all_case_mean = np.mean(acc_prkavg_arr, axis=0).tolist()
    a = [(t[0], t[1], j) for j, t in enumerate(all_case_mean)]
    a.sort(key=lambda x: (x[0], x[1]), reverse=True)
    headers = ['Case'] + ['PR@{}'.format(i+1) for i in range(10)] + ['PR@Avg', 'Acc', 'Time']
    for acc, prkavg, i in a[:1]:
        data = []
        for j in range(prks_arr.shape[0]):
    #         print_prk_acc(prks_arr[j, i, :], acc_arr[j, i])
            data.append([list(prks_dict.keys())[j], *prks_arr[j, i, :], np.mean(prks_arr[j, i, :]), 
                         acc_arr[j, i], time_arr[j][i]])
        data.append(['Avg', *np.mean([d[1:] for d in data], axis=0)])
        if verbose:
            print('{:-^80}'.format(f' Top @ {i} '))
            print(tabulate(data, headers=headers, floatfmt="#06.4f"))
    return a[0][2]

            
def print_topk_exp_results(rca_exp_res,
                           params=['mean_method', 'topk_path', 'prob_thres', 'num_sel_node'], 
                           n=5):
    print("Total Number of Exps:", len(rca_exp_res))
    l = []
    for i, d in enumerate(rca_exp_res):
        if d['acc'] is None:
            l.append((0, 0, i))
        else:
            l.append((d['acc'], np.mean(d['prks']), i))
    l.sort(reverse=True)
    for _, _, idx in l[:n]:
        print('@{} {:^80}'.format(idx, ','.join([f'{k}={rca_exp_res[idx][k]}' for k in params if k in rca_exp_res[idx]])))
        print_prk_acc(rca_exp_res[idx]['prks'], rca_exp_res[idx]['acc'])
        print("Total: {:.4f}".format(sum(rca_exp_res[idx]['time_info'].values())), 
              ', '.join([f'{k}: {v:.4f}' for k, v in rca_exp_res[idx]['time_info'].items()]))


def print_ranked_paths(ranked_paths, columns):
    for s, p in ranked_paths:
        print(f'{s:.2f}', end=': ')
        print(','.join([columns[n] for n in p]))


def print_ranked_nodes(ranked_nodes, columns):
    for n, s in ranked_nodes:
        print(f'{s:.2f} {columns[n]}')