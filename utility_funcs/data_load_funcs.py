import numpy as np
import pandas as pd
import datetime
import re


def interpolate_nan_values(df, missing_val=-1, method='linear'):
    """Interpolate missing values in DataFrame.
    Expect the input dataframe has a sorted index (0 to N or timestamps).
    Params:
        missing_val: default is -1. The missing values.
        method: default is linear. The interpolation method used. See pandas interpolate doc.
    """
    return df.replace(missing_val, np.nan).interpolate(method=method)


def normalize_df(df):
    data_mean = np.mean(df, axis=0)
    data_std = np.std(df, axis=0)
    df = (df - data_mean) / data_std
    df = df.fillna(0)
    return df


def get_numeric_df(df):
    from pandas.api.types import is_numeric_dtype
    inds = []
    print("Non numeric columns: ", end='')
    for i in range(df.shape[1]):
        if is_numeric_dtype(df.iloc[:, i]):
            inds.append(i)
        else:
            print(df.columns[i], end=', ')
    print()
    df_float = df.iloc[:, inds]
    return df_float

def get_nonconstant_inds(df, verbose=False):
    # Filter constant columns
    const_inds = []
    nonconst_inds = []
    for i in range(df.shape[1]):
        if len(df.iloc[:, i].unique()) <= 1:
            const_inds.append(i)
        else:
            nonconst_inds.append(i)
    if verbose:
        print("Num of non-constant columns: {}/{}".format(len(nonconst_inds), len(df.columns)))
        # print("\nNon-constant col inds:", nonconst_inds)
        print("Constant col names:", sorted([df.columns[i] for i in const_inds]))
    return nonconst_inds


def convert_timestamps(df, tz=None):
    '''
    Note: If tz is None, then datetime.datetime.fromtimestamp will use the system's timezone.
        Thus changing system's timezone will affect the datatime converted from the dataframe.
    '''
    if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        # already has timestamp index.
        print("skip converting timestamps")
        return df
    if 'timestamp' in df:
        # Parses timestamps
        t = []
        for d in df['timestamp']:
            dt = datetime.datetime.fromtimestamp(d, tz=tz)
            t.append(dt)
        t = np.array(t)

        # Sort dataframe by timestamps
        df_sorted = df.iloc[np.argsort(t), :]
        df_sorted = df_sorted.reset_index(drop=True)

        # Set the timestamps as index
        df_sorted['timestamp'] = np.sort(t)
        df_sorted = df_sorted.set_index('timestamp')
    elif df.index.name == 'timestamp':
        df.index = pd.to_datetime(df.index)
        df_sorted = df
    return df_sorted


def remove_dup_index(df):
    eq_inds = np.where(np.diff(np.sort(df.index)).astype(int) == 0)[0]
    print("eq_inds:", eq_inds)
    inds = list(range(len(df.index)))
    [inds.remove(i) for i in eq_inds]
    return df.iloc[inds, :]


def align_time_index(df, new_index=None, method='nearest', verbose=False):
    """Reindex to a new aligned time index, possiblly interpolate on missing index
    """
    if new_index is None:
        if verbose:
            print("New index: [{}, {}] with freq: {}".format(
                df.index[0], df.index[-1], df.index[1] - df.index[0]
            ))
        new_index = pd.date_range(start=df.index[0], end=df.index[-1],
                                  freq=df.index[1] - df.index[0])
    if verbose:
        print("Data length delta:", len(new_index) - df.shape[0])

    df = df.reindex(new_index, method=method)

    # Check the reindex results
#     print("Num of equal delta durations after aligning:", 
#           np.sum([int(i) == 30000000000 for i in np.diff(df.index)]))
    return df


def rename_df_columns(df_list, split_cols=[]):
    # Preprocess data, split by split_cols, and rename original columns
    if len(split_cols) == 0:
        return df_list
    df_list_new = []
    for df in df_list:
        if df.shape[0] == 0:
            assert False, "Empty dataframe in df_list"
        prefix = []
        stop = False
        for col in split_cols:
            if col not in df.columns:
                stop = True
                break
            prefix = prefix + [f"{col}={df[col][0]}"]
        if stop:
            print(f"Cannot find column {col} in input dataframe!")
            return df_list
        prefix = "(" + ",".join(prefix) + ")"
        cols_mapper = {}
        df = df.drop(split_cols, axis=1)
        for col in df.columns:
            cols_mapper[col] = f"{prefix}{col}"
        df = df.rename(cols_mapper, axis=1)
        df_list_new.append(df)
    return df_list_new


def split_df_by_col(df, col):
    dfs = []
    col_vals = []
    for s in df[col].unique():
        dfs.append(df[df[col] == s])
        col_vals.append(s)
    return dfs, col_vals


def load_preprocess_net_data(path, tz=None):
    df_net = pd.read_csv(path, index_col=0)
    df_net_splitbyiface, ifaces = split_df_by_col(df_net, 'iface')
    print(ifaces)
    # Timestamp convertion & Remove duplicate records
    df_net_splitbyiface = [convert_timestamps(i, tz=tz) for i in df_net_splitbyiface]
    df_net_splitbyiface = [remove_dup_index(i) for i in df_net_splitbyiface]
    # Timestamp alignment
    df_net_splitbyiface = [align_time_index(i) for i in df_net_splitbyiface]
    # Interpolate
    df_net_splitbyiface = [interpolate_nan_values(i) for i in df_net_splitbyiface]
    return df_net_splitbyiface


def load_preprocess_runtime_data(path, tz=None):
    if path.endswith('parquet'):
        df_runtime = pd.read_parquet(path)
    elif path.endswith('csv'):
        df_runtime = pd.read_csv(path, index_col=0)

    # Timestamp convertion
    df_runtime = convert_timestamps(df_runtime, tz=tz)
    df_runtime = remove_dup_index(df_runtime)

    # Timestamp alignment
    df_runtime = align_time_index(df_runtime)

    # drop columns with NA. Note, if we use interpolation, then dropping NA is not 
    # needed anymore.
    # df_runtime = df_runtime.dropna(axis=1, how='any')

    # Interpolate
    df_runtime = interpolate_nan_values(df_runtime)
    return df_runtime


def remove_mostly_nan_columns(df, most=0.8):
    counts = df.isna().sum(axis=0)
    drops = [c for c in df.columns if counts[c] >= most*df.shape[0]]
    print('Dropping', drops)
    return df.drop(drops, axis=1)


def apply_diff_df(df, col, verbose=False):
    if verbose:
        print(f'Apply 1st order differentiating to {col}')
    df = df.copy()
    if col in df:
        diff_arr = np.diff(df[col])
        diff_arr = np.concatenate([np.zeros([1]), diff_arr]) # prepend one 0
        df.insert(0, f"{col}_Diff", diff_arr)
        df = df.drop(col, axis=1)
    return df


def remove_col_parenthesis(s):
    res = re.match(r'\(.*\)(.*)', s)
    if res:
        return res.group(1)
    return s


def drop_mostna_columns(df, percentage=0.9):
    counts = df.isna().sum(axis=0)
    T = df.shape[0]
    for i in counts.index:
        print(f'{i:<60}: {counts[i]/T:7.2%} NAs', end='')
        if counts[i]/T >= percentage:
            print(' Dropped')
            df = df.drop(i, axis=1, errors='ignore')
        else:
            print()
    return df
