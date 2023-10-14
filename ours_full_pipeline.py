import os
import pickle
import pandas as pd
import argparse
# Import modified interval process functions
from dycause_lib.method_improves import (build_intervals_special, 
                                            generate_DCC,
                                            get_dir_hints_from_dcc,
                                            remove_double_direction_hints,
                                            global_thresholding,
                                            calculate_parcorr,
                                            calculate_GPDC,
                                            calculate_CMIknn
                                            )
import dycause_lib.method_improves as meth_imp
# Import data and graph plot functions
from utility_funcs.graph_draw import (get_node_colors,
                                          filter_dcc,
                                          filter_df,
                                          construct_d2g_map_dict,
                                          plot_graph_networkx_igraphlayout,
                                          mat_summary,
                                         )
from main_dycause_mp_new import dycause_causal_discover
from dycause_lib.rca import analyze_root, normalize_by_column
from utility_funcs.evaluation_function import pr_stat, print_prk_acc, my_acc
#------------- Domain Knowledge --------------------------------------
only_source_cols = ['inBytes', 'inMulticast', 'inPackets',
                    'tcpPkgInsegs', 'udpInDatagrams']
only_target_cols = ['outBytes', 'outPackets',
                    'tcpPkgOutsegs', 'tcpPkgRetranssegs',
                    'udpOutDatagrams']

def get_only_source_target_idx(only_source_cols, only_target_cols, col_list):
    def find_match_ind(simple_column):
        for i, c in enumerate(col_list):
            if c.find(simple_column) != -1:
                return i
#         assert False, f'Column {simple_column} not found in col_list!'
        return -1
    source_ids = [find_match_ind(i) for i in only_source_cols]
    source_ids = [i for i in source_ids if i!=-1]
    target_ids = [find_match_ind(i) for i in only_target_cols]
    target_ids = [i for i in target_ids if i != -1]
    return source_ids, target_ids


def main(df, all_columns, step, sign, lag, max_workers=15, max_segment_len=None, verbose=0,
         hints_thr=30, global_ntop=50, png_graph_name=None, 
         mean_method='arithmetic', topk_path=200, prob_thres=0.2, num_sel_node=3, out_of_path_nodes=True,
         use_cond=None, p_thr=0, comp_func=lambda x, p : x > p):
    local_results_dy, dcc_dy, mat_dy, time_stat_dict_dy = dycause_causal_discover(
        # Data params
        df.to_numpy()[:, :],
        # Granger interval based graph construction params
        step=step,
        significant_thres=sign,
        lag=lag,  # must satisfy: step > 3 * lag + 1
        adaptive_threshold=0.7,
        use_multiprocess=True,
        max_workers=max_workers,
        max_segment_len=max_segment_len,
        # Debug_params
        verbose=verbose,
        runtime_debug=True,
    )
    T, N = df.shape
    build_intervals_special(local_results_dy, sign, T, step, N)
    dcc_special = generate_DCC(local_results_dy, T, N, interval_key="intervals_special")
    data_idx_to_graph_idx, graph_idx_to_data_idx = construct_d2g_map_dict(
        list(df.columns), all_columns
    )
    
    filtered_dcc = filter_dcc(
        dcc_dy, len(all_columns), graph_idx_to_data_idx
    )
    filtered_dcc_special = filter_dcc(
        dcc_special, len(all_columns), graph_idx_to_data_idx
    )
    # hints from anomaly dcc
    hints = get_dir_hints_from_dcc(filtered_dcc, len(all_columns), hints_thr)
    o_s_ids, o_t_ids = get_only_source_target_idx(
        only_source_cols, only_target_cols, list(all_columns)
    )
    mat, candidates = global_thresholding(
        filtered_dcc_special,
        len(all_columns),
        normal_axis="none",
        ntop=global_ntop,
        only_source_ids=o_s_ids,
        only_target_ids=o_t_ids,
        hints=hints,
        return_candidates=True
    )
    import pdb; pdb.set_trace()
    filtered_df = filter_df(df, graph_idx_to_data_idx, mat.shape[0])
    mat_summary(mat)
    # Filter with partial correlation test results
    if use_cond == 'cmiknn120':
        cond_corr_res, cond_cmi_res = meth_imp.calculate_CMIknn_shuffled_mp(mat, filtered_df.iloc[-120:, :], max_workers=max_workers, verbose=True)
    elif use_cond == 'cmiknn':
        cond_corr_res, cond_cmi_res = meth_imp.calculate_CMIknn_shuffled_mp(mat, filtered_df, max_workers=max_workers, verbose=True)
    elif use_cond == 'parcorr':
        cond_corr_res = meth_imp.calculate_parcorr(mat, filtered_df)
    else:
        pass
    if use_cond in ['cmiknn120', 'cmiknn', 'parcorr']:
        mat = meth_imp.cond_test_filtering(mat, candidates, cond_corr_res, p_thr, comp_func)
    
    rca_results = {}
    for entry in range(len(all_columns)):
        ranked_nodes, out_path = analyze_root(
            normalize_by_column(mat),
            entry,
            filtered_df.to_numpy(),
            mean_method=mean_method,
            max_path_length=None,
            topk_path=topk_path,
            prob_thres=prob_thres,
            num_sel_node=num_sel_node,
            out_of_path_nodes=out_of_path_nodes,
            use_new_matrix=False,
            verbose=0,
        )
        # for n, s in ranked_nodes:
        #     print(all_columns[n], f'{s:.2f}')
        rca_results[entry] = {'ranked_nodes': ranked_nodes, 'out_path': out_path}
    return mat, rca_results


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='DyCause Root Cause Analysis.')
    # parser.add_argument('data', type=str, help='Input data frame. Shape: [T, N]')
    # parser.add_argument('step', type=int, help='DyCause step')
    # parser.add_argument('sign', type=float, help='DyCause significance value')
    # parser.add_argument('lag', type=int, help='DyCause lag')
    # parser.add_argument('max_workers', type=int, default=15, help='num of multiprocess workers')
    # parser.add_argument('max_segment_len', type=int, default=None, help='DyCause max_segment_len')
    # parser.add_argument('hints_thr', type=int, default=30, help='directional dcc hints threshold')
    # parser.add_argument('global_ntop', type=int, default=50,
    #     help='num of edges generated in global thresholding algorithm')
    # parser.add_argument('topk_path', type=int, default=200, help='num of backtrace paths')
    # parser.add_argument('prob_thres', type=float, default=0.2, help='minimum probility of backtrace paths')
    # parser.add_argument('num_sel_node', type=int, default=3, help='num of nodes selected in a path')
    # parser.add_argument('png_graph_name', type=str, default='/tmp/1.png', help='name of saved graph png')
    # parser.add_argument('verbose', type=int, default=0, help='verbose level')
    # args = parser.parse_args()
    df = pd.read_csv('sample_data/anomaly_host_metrics.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    # --------- Case 7 all_columns
    all_columns_d_3 = {
        "eth0": [
            "(iface=eth0)inPercent",
            "(iface=eth0)inPackets",
            "(iface=eth0)outPercent",
            "(iface=eth0)outPackets",
            "(iface=eth0)totalBytes",
            "(iface=eth0)totalPackets",
        ],
        "cpu": ["busy", "iowait", "softirq", "system", "user", "switches"],
        "kernel": ["kernelFilesAllocated"],
        "load": ["load1", "load5"],
        "memory": ["memBuffers", "memCached", "memShmem", "memUsedPercent", "memAvailablePercent"],
        "tcp": [
            "retrans",
            "tcpAbortOnTimeout",
            "tcpDelayedACKLocked",
            "tcpPkgInsegs",
            "tcpPkgOutsegs",
            "tcpTW",
        ],
        "socket": ["ssClosed", "ssEstab", "ssOrphaned", "ssTimeWait"],
        "udp": ["udpInDatagrams", "udpNoPorts", "udpOutDatagrams", "udpIgnoreMulti_Diff"],
        "disk": [
            "(mount=/data00)df.statistics.used.percent",
            "(mount=/data00)df.statistics.used",
            "(mount=/data00)df.statistics.total",
            "(mount=/data00)df.inodes.free.percent",
            "(mount=/data00)df.bytes.free.percent",
            "(mount=/)df.statistics.used.percent",
            "(mount=/)df.statistics.used",
            "(mount=/)df.statistics.total",
            "(mount=/)df.bytes.free.percent",
        ],
        "diskio": [
            "(device=sda)disk.io.write",
            "(device=sda)disk.io.w_wait",
            "(device=sda)disk.io.util",
            "(device=sda)disk.io.read",
            "(device=sda)disk.io.await",
            "(device=sda)disk.io.read_bytes_Diff",
            "(device=sda)disk.io.write_bytes_Diff",
            "(device=sdb)disk.io.write",
            "(device=sdb)disk.io.w_wait",
            "(device=sdb)disk.io.util",
            "(device=sdb)disk.io.read",
            "(device=sdb)disk.io.await",
            "(device=sdb)disk.io.read_bytes_Diff",
            "(device=sdb)disk.io.write_bytes_Diff",
        ],
    }

    def extract_cols(d):
        all_columns = []
        for k, v in d.items():
            all_columns += sorted(v)
        return all_columns

    all_columns = extract_cols(all_columns_d_3)
    print("Num of all columns:", len(all_columns))
    
    mat, rca_results = main(
        df, all_columns, 30, 0.05, 7, max_workers=15, max_segment_len=None, verbose=2,
        hints_thr=30, global_ntop=80, png_graph_name=None, 
        mean_method='arithmetic', topk_path=400, prob_thres=0.6, num_sel_node=5, out_of_path_nodes=True,
        use_cond='cmiknn120', p_thr=0.7)
    entry = all_columns.index('load1')
    root_cause_list = [all_columns.index(i) for i in ["memBuffers", "memCached", "memShmem", "memUsedPercent", "memAvailablePercent"]]
    prks = pr_stat(rca_results[entry]['ranked_nodes'], root_cause_list, 10)
    acc = my_acc(rca_results[entry]['ranked_nodes'], root_cause_list, n=len(all_columns))
    print_prk_acc(prks, acc)
    import pdb; pdb.set_trace()