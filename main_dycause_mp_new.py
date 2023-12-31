import datetime
import os
import pickle
import random
import time
import numpy as np
import pandas as pd

from dycause_lib.temporal_analyze import TemporalAnalyze
from dycause_lib.causal_graph_build import adaptive_thresholding

def dycause_causal_discover(
    # Data params
    data,
    # Granger interval based graph construction params
    step=60,
    significant_thres=0.1,
    lag=10,  # must satisfy: step > 3 * lag + 1
    adaptive_threshold=0.7,
    use_multiprocess=True,
    max_workers=3,
    mp_mode=2,
    opt_method="fast_version_3",
    max_segment_len=None,
    # Debug_params
    verbose=False,
    runtime_debug=False,
    *args,
    **kws,
):
    """DyCause Causal Discover Algorithm

    Args:
        data (numpy array): The input time series of shape [N, T]
        step (int, basic window size): The basic window size. Defaults to 60.
        significant_thres (float, optional): Granger significance value. Defaults to 0.1.
        lag (int, optional): How many past values from time t. Defaults to 10.
        use_multiprocess (bool, optional): Whether use multiprocess library. 
                                           If False, use multithread library. Defaults to True.
        max_workers (int, optional): Maximum process or thread number. Defaults to 3.
        opt_method (str, optional): Which optimization method ('call_package', 'standard', 'fast_version_3').
                                    'call_package': Granger test from statsmodels
                                    'standard': Manually implemented Granger test
                                    'fast_version_3': Paper Granger causal interval version.
        max_segment_len: the maximum tested sliding window. Default is None, which means the maximum is all the data.
        verbose (bool, optional): Whether print runtime info. Defaults to False.
        runtime_debug (bool, optional): Whether enable run time test mode. 
                                        This is used to measure run time. Defaults to False.
    """
    data_source = "temp"
    np.random.seed(42)
    random.seed(42)
    if runtime_debug:
        time_stat_dict = {}
        tic = time.time()
    if verbose:
        # verbose level >= 1: print method name
        print("{:-^80}".format("DyCause"))
    dir_output = "temp_results/dycause/" + data_source
    os.makedirs(dir_output, exist_ok=True)
    # region Run loop_granger to get the all causal intervals
    local_length = data.shape[0]
    if max_segment_len is None:
        max_segment_len = local_length
    ta_inst = TemporalAnalyze(
        dir_output,
        step,
        lag,
        significant_thres,
        max_segment_len,
        step,
        method=opt_method,
        verbose=verbose,
        runtime_debug=runtime_debug,
    )
    ta_inst.granger_analyze(
        data,
        use_multiprocess=use_multiprocess,
        mp_mode=mp_mode,
        max_workers=max_workers
    )

    # Construction dynamic causality curves (DCC) using generated intervals
    ta_inst.generate_DCC(local_length, data.shape[1])
    # Build the dependency graph from DCC.
    trans_mat = adaptive_thresholding(
        ta_inst.DCC, adaptive_threshold, data.shape[1]
    )
    if runtime_debug:
        toc = time.time()
        time_stat_dict["Construct-Impact-Graph-Phase"] = toc - tic
        tic = toc
        
    if runtime_debug:
        # Update the runtime info from TemporalAnalyze class.
        time_stat_dict.update(ta_inst.time_stat_dict)
        # Use the timezone in my location.
        # local_tz = datetime.timezone(datetime.timedelta(hours=8))
        # time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
        # if verbose:
        #     print("{:<10}".format("") + "Saving runtime data to " + f"time_stat_dict_{time_str}.pkl")
        # safe_dump_obj(time_stat_dict, os.path.join(dir_output,"runtime-data",f"time_stat_dict_{time_str}.pkl"))

    # endregion
    if not runtime_debug:
        return ta_inst.local_results, ta_inst.DCC, trans_mat
    else:
        return ta_inst.local_results, ta_inst.DCC, trans_mat, time_stat_dict


def normalize_df(df):
    data_mean = np.mean(df, axis=0)
    data_std = np.std(df, axis=0)
    df = (df - data_mean) / data_std
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    # Load data
    # with open("temp_results/tmp_data/dycause_input_host139_runtime.pkl", "rb") as f:
    #     dycause_input = pickle.load(f)
    # d = "processed_data/host197"
    # df = pd.read_csv("{}/selected_merged_data.csv".format(d), index_col=0)
    
    # Host 198
    data_out_dir = "processed_data/host198"
    df = pd.read_csv("{}/dycause_input.csv".format(data_out_dir), index_col=0)

    # Host 24 and 151
    # data_out_dir = "processed_data/host24and151"
    # df = pd.read_csv("{}/selected_merged_data.csv".format(data_out_dir), index_col=0)

    # Host 204 and 28
    # data_out_dir = "processed_data/host204and28"
    # df = pd.read_csv("{}/selected_merged_data.csv".format(data_out_dir), index_col=0)

    print("Data shape:", df.shape)

    df = normalize_df(df)

    sign = 0.1
    step = 200
    lag = 5
    max_segment_len=None
    local_results_dy, dcc_dy, mat_dy, time_stat_dict_dy = dycause_causal_discover(
        # Data params
        df.to_numpy()[:, :],
        # Granger interval based graph construction params
        step=step,
        significant_thres=sign,
        lag=lag,  # must satisfy: step > 3 * lag + 1
        adaptive_threshold=0.7,
        use_multiprocess=True,
        max_workers=3,
        opt_method="fast_version_3",
        max_segment_len=max_segment_len,
        # Debug_params
        verbose=2,
        runtime_debug=True,
    )

    print(time_stat_dict_dy['Construct-Impact-Graph-Phase'])

    # Use the timezone in my location.
    local_tz = datetime.timezone(datetime.timedelta(hours=8))
    time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    fname = os.path.join("temp_results", "host198", f"exp_rets_{time_str}.pkl")
    print("Saving results to:", fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump({
            "local_results": local_results_dy,
            "dcc": dcc_dy,
            "mat": mat_dy,
            "time_stat_dict": time_stat_dict_dy
        }, f)
