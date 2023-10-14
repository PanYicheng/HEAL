import matplotlib.pyplot as plt
import os
import re
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from utility_funcs.data_load_funcs import normalize_df
from dycause_lib.causal_graph_build import get_segment_split


def draw_weighted_graph(transition_matrix, filename, weight_multiplier=4):
    """Draw weighted graph given transition probability
    
    The parameter is a transition matrix which the i,j element
    represent the probability of transition from i to j
    The weight is illustrated using edge width
    """
    n = len(transition_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    g = nx.MultiDiGraph()
    edge = []
    edge_weight = dict()
    pos = dict()
    for i in range(1, n+1):
        angle = (i+0.0)/n*(2*np.pi)
        pos[i] = (np.cos(angle), np.sin(angle))
    g.clear()
    ax.clear()
    for x_i in range(1, n+1):
        for y_i in range(1, n+1):
            if transition_matrix[x_i-1][y_i-1] > 0:
                g.add_edge(x_i, y_i)
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = transition_matrix[x_i-1][y_i-1]
    # No edge, just return
    if len(edge) == 0:
        return
    nx.draw_networkx(g, pos, with_labels=True, ax=ax, node_size=1000, font_size=25,
                     font_color='y', node_color='k', arrows=False)
    weight_list = np.array([edge_weight[key] for key in g.edges()])
    weight_list = weight_list / np.max(weight_list) * weight_multiplier
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges(), alpha=0.6,
                           width=weight_list, arrowsize=35,
                           connectionstyle='arc3,rad=0')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create directory if not exists
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)
    # plt.show()


def plot_data(df, cols, t_slice=None, use_one_axe=False, normalize_data=False, plot_type='scatter'):
    if t_slice == None:
        t_slice = slice(None, None)

    def normalize_ts(ts):
        if not normalize_data:
            return ts
        return (ts -ts.mean()) / ts.std()

    toppad = 0.2
    if use_one_axe:
        h = 2 + toppad
    else:
        h = len(cols) * 1 + toppad
    fig = plt.figure(figsize=[8, h])
    axs = fig.subplots(1 if use_one_axe else len(cols), 1, sharex=True, squeeze=False)
    # fig.subplots_adjust(top= (h - toppad) / h, hspace=0.1)
    fig.subplots_adjust(top=1.0, left=0, right=1.0, bottom=0, hspace=0.35, wspace=0)
    if normalize_data:
        fig.suptitle('Normalized Data', x=0.5, y=(h - toppad) / h, ha='center', va='bottom')
    for i, n in enumerate(cols):
        x = df.loc[t_slice].index
        if use_one_axe:
            ax = axs[0, 0]
        else:
            ax = axs[i, 0]
        if plot_type == 'scatter':
            ax.scatter(x, normalize_ts(df.loc[t_slice][n]), label=n, alpha=0.8 if use_one_axe else 1.0, s=1, color='k')
        else:
            ax.plot(x, normalize_ts(df.loc[t_slice][n]), lw=1, label=n, alpha=0.8 if use_one_axe else 1.0)
    #     axs[i, 0].set_yscale("log")
    #     axs[i, 0].set_title(f"{n}")
    #     axs[i, 0].set_ylabel("GiB")
        
        
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.EngFormatter(sep=''))
        # plt.setp(ax.get_xticklabels(), fontsize='xx-large')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.setp(ax.get_yticklabels(), fontsize='xx-large')
        ax.set_title(n, pad=3, y=1.0)
        # ax.grid(axis='both', color='gray')
        ax.tick_params(labelsize='small')
    import matplotlib.dates as mdates
    # locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.AutoDateFormatter(locator)
    # formatter = mdates.ConciseDateFormatter(locator)
    formatter.scaled[1/(24*60)] = '%H:%M' # only show hour and min
    ax.xaxis.set_major_formatter(formatter)
    # [ax.legend(loc='upper left') for ax in axs.flatten()]
    # fig.tight_layout()
    # plt.show()


def plot_all_data(df, savepath, title=None, tick_locator=None):
    if df.shape[0] == 0 or df.shape[1] == 0:
        print("Empty dataframe to plot!")
        return
    toppad = 0.5
    h = df.shape[1] * 2 + toppad
    fig = plt.figure(figsize=[14, h])
    axs = fig.subplots(df.shape[1], 1, sharex=True, squeeze=False)
    fig.subplots_adjust(top=df.shape[1]*2/ h)
    if title is not None:
        fig.suptitle(title, x=0, y=1.0, ha='left', va='top', weight='bold')
    x = df.index
    for i in range(df.shape[1]):
        label=df.columns[i]
        axs[i, 0].plot(x, df.iloc[:, i])
        axs[i, 0].text(0, 1, label, ha='left', va='top', transform=axs[i, 0].transAxes,
                       fontsize='large', weight='bold')
        axs[i, 0].grid(axis='both')
        import matplotlib.dates as mdates
        if tick_locator is None:
            tick_locator = mdates.AutoDateLocator(minticks=10)
        axs[i, 0].xaxis.set_major_locator(tick_locator)
        axs[i, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(tick_locator))
        axs[i, 0].tick_params(labelbottom=True)
    fig.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_legend_colorbar(type_color_d, cmap, vmin, vmax, ncol=6, savepath=None):
    '''Plot the legend and the colorbar for the igraph plot (igraph has no such things.)
    '''
    fig = plt.figure(figsize=[12, 2])
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                        wspace=0, hspace=0,
                        width_ratios=[1], height_ratios=[5, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.set_frame_on(False)
    ax1.set_axis_off()

    # Draw colorbar
    import matplotlib as mpl
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax2, orientation='horizontal', label='Edge weight')

    # Draw legends
    from matplotlib.patches import Circle
    legend_lines = []
    for t, c in type_color_d.items():
        legend_lines.append(Circle((0, 0), color=c, label=t))
    if len(legend_lines)>0:
        ax1.legend(handles=legend_lines, loc='center', ncol=ncol, fontsize='x-large')

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def remove_parenthesis(s):
    res = re.match(r'\(.*\)(.*)', s)
    if res:
        return res.group(1)
    return s

    
def get_node_colors(columns_d, cols, cmap_name="Paired", verbose=False):
    types = list(columns_d.keys())
    if verbose:
        print(types)
    ncolor = len(types)
    label_cmap = plt.get_cmap(cmap_name)
    def get_color(t):
        return label_cmap(types.index(t) / ncolor)
    type_color_d = {}
    for t in types:
        type_color_d[t] = get_color(t)
    def col2color(col):
        for k,v in columns_d.items():
            for c in v:
                if remove_parenthesis(col) == remove_parenthesis(c):
                    return get_color(k)
    node_colors = []
    for col in cols:
        node_colors.append(col2color(col))
    return type_color_d, node_colors
    

def plot_two_dcc(exp_ret, df, pair, graph_idx_to_data_idx=None, savepath=None):
    if graph_idx_to_data_idx is None:
        graph_idx_to_data_idx = {}
        for i in range(df.shape[1]):
            graph_idx_to_data_idx[i] = i
    # zoom_l, zoom_r = 40, 120
    fig = plt.figure(figsize=[6, 4])
    axs = fig.subplots(2, 1, sharex=True)
    fig.tight_layout()
    df = normalize_df(df)
    pair_data = [graph_idx_to_data_idx[_] for _ in pair]
    axs[0].plot(df.index, df.iloc[:, pair_data[0]], label=df.columns[pair_data[0]], alpha=1.0)
    axs[0].plot(df.index, df.iloc[:, pair_data[1]], label=df.columns[pair_data[1]], alpha=1.0)
    axs[0].set_title("Normalized Data")
    i, j = pair_data
    def two_dcc(ax, dcc, i, j, pref=''):
        for _i, _j in [(i, j), (j, i)]:
            s = float(np.sum(dcc[f'{_i}->{_j}']))
            ax.plot(df.index, dcc[f'{_i}->{_j}'], 
                    label=f'{pref}{df.columns[_i]}->{df.columns[_j]} {s}', alpha=0.9)
    two_dcc(axs[1], exp_ret['dcc'], i, j)
    # two_dcc(axs[2], exp_ret["dcc_special"], i, j, pref="S ")

    # segs = get_segment_split(df.shape[0], exp_ret['step'])
    # def add_seg_line(ax, segs):
    #     for seg in segs[:-1]:
    #         ax.axvline(df.index[seg], color='r', ls='--', alpha=0.3)
    #         ax.text(df.index[seg], 0.1, f'{seg}', transform=ax.get_xaxis_transform(), color='r', alpha=0.7)
    # add_seg_line(axs[1], segs)
    # add_seg_line(axs[2], segs)

    # axs[3].set_xlim(df.index[zoom_l], df.index[zoom_r])
    # axs[0].set_ylim(np.min(df.iloc[zoom_l:zoom_r, 0]), np.max(df.iloc[zoom_l:zoom_r, 0]))
    # axs[1].set_ylim(np.min(df.iloc[zoom_l:zoom_r, 1]), np.max(df.iloc[zoom_l:zoom_r, 1]))

    import matplotlib.dates as mdates
    locator = mdates.MinuteLocator(byminute=range(0, 60, 15))
    axs[1].xaxis.set_major_locator(locator)
    formatter = mdates.AutoDateFormatter(locator)
    formatter.scaled[1/(24*60)] = '%H:%M' # only show hour and min
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].set_ylabel("DCS Strength")
    axs[1].set_xlabel("Time")
    # axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    [ax.legend(loc='upper left') for ax in axs]
    # axs.legend(loc='upper right')
    
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    # plt.show()


def filter_dcc(dcc, N_new, graph_idx_to_data_idx):
    # Filter dcc and transition matrix
    filtered_dcc = {}
    all_zero_dcc = np.zeros_like(next(iter(dcc.values())))
    for i in range(N_new):
        for j in range(N_new):
            try:
                i_old = graph_idx_to_data_idx[i]
                j_old = graph_idx_to_data_idx[j]
                filtered_dcc[f'{i}->{j}'] = dcc[f'{i_old}->{j_old}']
            except KeyError as e:
                filtered_dcc[f'{i}->{j}'] = all_zero_dcc
    return filtered_dcc


def filter_df(ori_df, graph_idx_to_data_idx, N):
    filtered_df = {}
    for i in range(N):
        filtered_df[i] = ori_df.iloc[:, graph_idx_to_data_idx[i]]
    filtered_df = pd.DataFrame(filtered_df)
    return filtered_df


def filter_hints(hints, data_idx_to_graph_idx):
    new_hints = []
    for i, j, v in hints:
        if i in data_idx_to_graph_idx and j in data_idx_to_graph_idx:
            new_hints.append((data_idx_to_graph_idx[i], data_idx_to_graph_idx[j], v))
    return new_hints


def construct_d2g_map_dict(data_cols, graph_cols, print_info=False):
    """Construct a map that translate data index to graph index and vice versa.
    Params:
        print_info: whether to print the detailed translate info.
    """
    # Map column names
    graph_idx_to_data_idx = {}
    for i, n in enumerate(graph_cols):
        if n in data_cols:
            graph_idx_to_data_idx[i] = data_cols.index(n)

    data_idx_to_graph_idx = {}
    for k, v in graph_idx_to_data_idx.items():
        data_idx_to_graph_idx[v] = k

    if print_info:
        for i, n in enumerate(graph_cols):
            print("Graph.{:<2}: {:<25} <-- ".format(i, n), end='')
            if i in graph_idx_to_data_idx:
                print("Data.{:<2}".format(graph_idx_to_data_idx[i]))
        #         print("{} ({:.2f})".format(from_anomaly_map[i], high_freq_list[from_anomaly_map[i]]))
            else:
                print("None")
    return data_idx_to_graph_idx, graph_idx_to_data_idx


def plot_igraph(mat, node_colors, backend="default", savename=None,
                vertex_label_color=None):
#     mat = transition_matrix
#     backend="default"
    g = igraph.Graph(directed=True)
    N = mat.shape[0]
    assert N==len(node_colors), "Num of node_colors should be size of mat."
    g.add_vertices(N)

    edges = []
    ws = []
    ecolors = []
    cmap = plt.get_cmap('viridis')

    for i in range(N):
        for j in range(N):
            if mat[i, j]>0 and i!=j:
                edges.append((i, j))
                ws.append(mat[i, j])
    # Normalize ws to [0, 1]
    ws = np.array(ws)
    ws1_max = np.max(ws)
    ws = list(ws / np.max(ws))
    for i in ws:
        ecolors.append(cmap(i))

#     # Special Edges
#     edges_ = []
#     ws_ = []
#     for i, j, v in special_edges[:150]:
#         edges_.append((i, j))
#         ws_.append(v)
#     #     ecolors.append((0.5, 0.5, 0.5))
#     # Normalize ws to [0, 1]
#     ws_ = np.array(ws_)
#     ws2_max = np.max(ws_)
#     ws_ = list(ws_ / np.max(ws_))
#     for i in ws_:
#         ecolors.append(cmap(i))

    # Merge edges
#     edges.extend(edges_)
#     ws.extend(ws_)

    g.add_edges(edges)
    g.es['color'] = ecolors
    # g.es['curved'] = [0.1 for i in range(len(edges))]
    g.es['weight'] = ws
    g.es['width'] = 3
    igraph.summary(g)
    print(g.degree(mode="in"))
    
    init_node_pos = []
    for i in range(N):
        angle = i/N * 2 * np.pi
        init_node_pos.append([np.cos(angle), np.sin(angle)])
    init_node_pos = np.array(init_node_pos)
    l_name = "fruchterman_reingold"
    layout = g.layout(l_name, weights='weight', niter=0, start_temp=0.05, seed=init_node_pos)
#     l_name = "reingold_tilford"
#     l_name = 'circle'
#     layout = g.layout('kk')
#     layout = g.layout_circle()
    # g.vs["name"] = plot_columns

    # num_plot_edges = 75
    # g.add_edges(edges[:num_plot_edges])
    # g.add_edges(edges[75:75+5])

    if backend == "matplotlib":
        fig = plt.figure(figsize=[10, 10])
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_frame_on(False)
        ax.set_axis_off()

    visual_style = {}
    visual_style["vertex_size"] = 40
    visual_style["vertex_color"] = node_colors
    visual_style["vertex_label_size"] = 25
#     visual_style["vertex_label_angle"] = [30/180*np.pi for i in range(N)]
    visual_style["vertex_label"] = [str(i) for i in range(N)]
    if vertex_label_color is not None:
        visual_style["vertex_label_color"] = vertex_label_color
    # visual_style["vertex_label_dist"] = [-1 for i in range(N)]
#     width_scale = 2.0 / np.max(ws)
    # visual_style["edge_color"] = ecolors
    visual_style["edge_arrow_size"] = 1.0
    visual_style["edge_arrow_width"] = 1.0
    visual_style["edge_curved"] = [0 for i in range(len(ws))]
    visual_style["layout"] = layout
    visual_style["bbox"] = (1000, 1000)
    # visual_style["margin"] = 20
#     visual_style["autocurve"] = True
    if backend == "matplotlib":
        visual_style["target"] = ax
        if savename is not None:
            igraph.plot(g, **visual_style)
            plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            return igraph.plot(g, **visual_style)
    else:
        if savename is not None:
            return igraph.plot(g, target=savename, **visual_style)
        else:
            return igraph.plot(g, **visual_style)


def plot_graph_networkx(mat, node_colors, savename=None):
    fig = plt.figure(figsize=[10, 10])
    gs = gridspec.GridSpec(1, 1, figure=fig, 
                           wspace=0, hspace=0,
                           width_ratios=[1], height_ratios=[1])
    ax = fig.add_subplot(gs[0, 0])

    N = mat.shape[0]
    nodes = list(range(N))

    g = nx.DiGraph()
    for i in range(N):
        g.add_node(i)
        for j in range(N):
            if mat[i, j]>0 and i!=j:
                g.add_edge(i, j, weight=mat[i, j])

    # Graph is not planar, spectral_layout many nodes are in the same position,
    from networkx.drawing.layout import kamada_kawai_layout, planar_layout, shell_layout, \
                                        spring_layout, spectral_layout, spiral_layout, circular_layout
    node_pos = spring_layout(g, k=5, weight='weight', iterations=50, seed=42)
    # node_pos = spiral_layout(g, resolution=5)
#     node_pos = circular_layout(g)

    node_size=600

    nx.draw_networkx_labels(g, node_pos, labels={i: str(i) for i in nodes}, ax=ax)
    # node_colors = ['c' if n<=65 else 'g' for n in nodes]
    nx.draw_networkx_nodes(g, node_pos, alpha=1.0, ax=ax, node_shape='H',
                           nodelist=nodes, node_color=node_colors,
                           node_size=node_size)

    cmap = plt.get_cmap('viridis')
    # cmap = plt.get_cmap("seismic")
    cs = []
    for (i, j) in g.edges():
        cs.append(float(mat[i, j]))
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=np.max(cs))
    # norm = Normalize()
    cs = norm(cs)
    cs = [cmap(i) for i in cs]
    #     print(cs)

    edgelist = g.edges

    #     from IPython.core.debugger import set_trace
    #     set_trace() # 断点位置
    patches = nx.draw_networkx_edges(g, node_pos, ax=ax,
        arrows=True, arrowstyle='->', arrowsize=10, 
#         connectionstyle="angle, angleA=90, angleB=0, rad=3",
#         connectionstyle="arc3, rad=0.5",
        alpha=0.8,
        edgelist=edgelist, 
        edge_color=cs,
        width=2, node_size=node_size)
    ax.set_frame_on(False)

    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_graph_networkx_igraphlayout(mat, node_colors, savename=None, 
                                     edge_w_norm='global', layout_meth='default', ax=None):
    if ax is None:
        fig = plt.figure(figsize=[10, 10])
        gs = gridspec.GridSpec(1, 1, figure=fig, 
                               wspace=0, hspace=0,
                               width_ratios=[1], height_ratios=[1])
        ax = fig.add_subplot(gs[0, 0])

    N = mat.shape[0]
    nodes = list(range(N))

    g = nx.DiGraph()
    g_igraph = igraph.Graph(directed=True)
    g_igraph.add_vertices(N)
    edges = []
    ws = []
    for i in range(N):
        g.add_node(i)
        for j in range(N):
            if mat[i, j] > 0:
                edges.append((i, j))
                ws.append(mat[i, j])
                g.add_edge(i, j)
    if edge_w_norm == 'global':
    # Normalize ws to [0, 1]
        ws = np.array(ws)
        ws = list(ws / np.max(ws))
    elif edge_w_norm == 'none':
        pass
    g_igraph.add_edges(edges)
    g_igraph.es['weight'] = ws
#     igraph.summary(g_igraph)
#     print(g_igraph.degree(mode="in"))
    
    init_node_pos = []
    for i in range(N):
        angle = i/N * 2 * np.pi
        init_node_pos.append([np.cos(angle), np.sin(angle)])
    init_node_pos = np.array(init_node_pos)
    if layout_meth=='freingold':
        l_name = "fruchterman_reingold" #
        layout = g_igraph.layout(l_name, 
            weights='weight', niter=400, start_temp=0.05, seed=init_node_pos
        )
        layout.mirror(1) # to keep the same look of networkx and igraph plot
        node_pos = {i: pos for i, pos in enumerate(layout.coords)}
    else:
        node_pos = {i: pos for i, pos in enumerate(init_node_pos)}
#     l_name = "reingold_tilford"
#     l_name = 'circle'
#     layout = g_igraph.layout('kk')
#     layout = g_igraph.layout_circle()

    node_size=600

    nx.draw_networkx_labels(g, node_pos, labels={i: str(i) for i in nodes}, ax=ax)
    nodes_patches = nx.draw_networkx_nodes(g, node_pos, alpha=1.0, ax=ax, node_shape='H',
                           nodelist=nodes, node_color=node_colors,
                           node_size=node_size)

    cmap = plt.get_cmap('viridis')
    e_colors = []
    for w in ws:
        e_colors.append(cmap(w))
    #     from IPython.core.debugger import set_trace
    #     set_trace() # 断点位置
    # Note: use edges to keep the same edge orders, DO NOT USE g.edges
    edges_patches = nx.draw_networkx_edges(g, node_pos, ax=ax,
        arrows=True, arrowstyle='->', arrowsize=10, 
#         connectionstyle="angle, angleA=90, angleB=0, rad=3",
#         connectionstyle="arc3, rad=0.5",
        alpha=0.8,
        edgelist=edges, 
        edge_color=e_colors,
        width=np.array(ws) * 5.0, node_size=node_size)
    ax.set_frame_on(False)
    # ax.set_xlim(-1.2, 1.2)
    # ax.set_ylim(-1.2, 1.2)

    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.close()
    return nodes_patches, edges_patches


def plot_graph_networkx_igraphlayout_withlabel(mat, node_colors, correct_edges, wrong_edges,
                                               savename=None, edge_w_norm='global', layout_meth='default',
                                               root_causes=None, entry=None):
    fig = plt.figure(figsize=[10, 10])
    gs = gridspec.GridSpec(1, 1, figure=fig, 
                           wspace=0, hspace=0,
                           width_ratios=[1], height_ratios=[1])
    ax = fig.add_subplot(gs[0, 0])

    N = mat.shape[0]
    nodes = list(range(N))

    g = nx.DiGraph()
    g_igraph = igraph.Graph(directed=True)
    g_igraph.add_vertices(N)
    edges = []
    ws = []
    for i in range(N):
        g.add_node(i)
        for j in range(N):
            if mat[i, j]>0:
                edges.append((i, j))
                ws.append(mat[i, j])
                g.add_edge(i, j)
    if edge_w_norm == 'global':
    # Normalize ws to [0, 1]
        ws = np.array(ws)
        ws = list(ws / np.max(ws))
    elif edge_w_norm == 'none':
        pass
    g_igraph.add_edges(edges)
    g_igraph.es['weight'] = ws
#     igraph.summary(g_igraph)
#     print(g_igraph.degree(mode="in"))
    
    init_node_pos = []
    for i in range(N):
        angle = i/N * 2 * np.pi
        init_node_pos.append([np.cos(angle), np.sin(angle)])
    init_node_pos = np.array(init_node_pos)
    if layout_meth=='freingold':
        l_name = "fruchterman_reingold" #
        layout = g_igraph.layout(l_name, 
            weights='weight', niter=400, start_temp=0.05, seed=init_node_pos
        )
        layout.mirror(1) # to keep the same look of networkx and igraph plot
        node_pos = {i: pos for i, pos in enumerate(layout.coords)}
    else:
        node_pos = {i: pos for i, pos in enumerate(init_node_pos)}
#     l_name = "reingold_tilford"
#     l_name = 'circle'
#     layout = g_igraph.layout('kk')
#     layout = g_igraph.layout_circle()

    node_size=600

    nx.draw_networkx_labels(g, node_pos, labels={i: str(i) for i in nodes}, ax=ax)
    edgecolors = ['gray' for i in range(N)]
    linewidths = [0 for i in range(N)]
    if root_causes is not None:
        for i in root_causes:
            edgecolors[i] = 'r'
            linewidths[i] = 3
    if entry is not None:
        edgecolors[entry] = '#f0f00c'
        linewidths[entry] = 3
    nodes_patches = nx.draw_networkx_nodes(g, node_pos, alpha=1.0, ax=ax, node_shape='H',
                                           nodelist=nodes, node_color=node_colors, edgecolors=edgecolors,
                                           linewidths=linewidths,
                                           node_size=node_size)

    cmap = plt.get_cmap('viridis')
    e_colors = []
    if correct_edges is None:
        for w in ws:
            e_colors.append(cmap(w))
    else:
        for e, w in zip(edges, ws):
            if e in correct_edges:
                e_colors.append('g')
            elif e in wrong_edges:
                e_colors.append('r')
            else:
                e_colors.append('k')
    #  from IPython.core.debugger import set_trace
    #  set_trace() # 断点位置
    # Note: use edges to keep the same edge orders, DO NOT USE g.edges
    edges_patches = nx.draw_networkx_edges(g, node_pos, ax=ax,
        arrows=True, arrowstyle='->', arrowsize=10, 
#         connectionstyle="angle, angleA=90, angleB=0, rad=3",
#         connectionstyle="arc3, rad=0.5",
        alpha=0.8,
        edgelist=edges, 
        edge_color=e_colors,
        width=np.array(ws) * 5.0, node_size=node_size)
    ax.set_frame_on(False)
    # ax.set_xlim(-1.2, 1.2)
    # ax.set_ylim(-1.2, 1.2)

    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.close()
    return nodes_patches, edges_patches


def mat_summary(mat):
    print(f"Num of nonzeros: {np.sum(mat > 0)}, min={np.min(mat)}, max={np.max(mat)}")
    

def print_edge_details(edges, cols):
    for e in edges:
        v = None
        if len(e) == 2:
            i, j = e
        elif len(e) == 3:
            i, j, v = e
        else:
            print("Invalid edges parameters.")
            return
        print(f"{i:<2} {cols[i]:<25} -> {j:<2} {cols[j]:<25} {v}")
