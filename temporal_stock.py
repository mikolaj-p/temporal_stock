import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
from os import listdir
from os.path import isfile, join
import time
import random
import itertools

def plot_stats(timestamps, stats, label, file_name):
    plt.figure(figsize=(8,6))
    plt.figure()
    plt.plot(timestamps, stats, 'r-', label=label)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_stats_set(timestamps, stats_set, file_name):
    plt.figure(figsize=(8,6))
    colormap = plt.cm.gist_rainbow(np.linspace(0, 1, 5))
    l_styles = ['-','--','-.',':']
    for (label, stat), (linestyle, color) in zip(stats_set.items(), itertools.product(l_styles, colormap)):
        plt.plot(timestamps, stat, color=color, linestyle=linestyle, label=label)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def calc_heterogeneity(G):
    N = G.number_of_nodes()
    return (N - 2*(sum([1/math.sqrt(G.degree[e[0]]*G.degree[e[1]]) for e in G.edges])))/(N - 2*math.sqrt(N-1))

def create_min_st(instruments, corr_dict, instr_max):
    G = nx.Graph()
    for instr in instruments[:instr_max]:
        G.add_node(instr)

    for edge, weight in corr_dict.items():
        G.add_edge(*edge, weight=weight)

    return nx.minimum_spanning_tree(G, weight='weight')

def create_max_st(instruments, corr_dict, instr_max):
    G = nx.Graph()
    for instr in instruments[:instr_max]:
        G.add_node(instr)

    for edge, weight in corr_dict.items():
        G.add_edge(*edge, weight=weight)

    return nx.maximum_spanning_tree(G, weight='weight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, default=300, help='Window length')
    parser.add_argument('--shift', type=int, default=25, help='Shift size')
    parser.add_argument('--instr-max', type=int, default=470, help='Maximum number of instruments used in graph construction')
    parser.add_argument('--data-dir', type=str, default='stocks', help='Directory containing the data')
    parser.add_argument('--min', action='store_true', help='Use minimal spanning tree (default: use maximal spanning tree)')
    parser.add_argument('--plot-graphs', action='store_true', help='Plot all graphs in temporal network')
    args = parser.parse_args()

    print("Loading data", end=' ')
    start_t = time.time()
    instruments = [f.split("_")[0] for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    if args.instr_max < len(instruments):
        random.shuffle(instruments)
    else:
        instruments = sorted(instruments)

    df_log_ret = pd.DataFrame()
    df_timestamps = None
    for instr in instruments[:args.instr_max]:
        df = pd.read_csv(os.path.join(args.data_dir, instr+'_data.csv'), sep=',', parse_dates=['date'], dtype={'close': np.float32})
        if df_timestamps is None:
            df_timestamps = df['date']
        df_log_ret[instr] = df['close'].rolling(2).apply(lambda x: math.log(x[1]) - math.log(x[0]), raw=True)
    print("%.2f s"%(time.time() - start_t))

    print("Creating graphs")
    timestamps = []
    paths_weighted = []
    paths_unweighted = []
    heterogeneity = []
    degrees_list = []
    leaves_nbr = []

    for i in range(args.delta+1, len(df_log_ret.index), args.shift):
        print("Step " + str(i), end=' ')
        step_start_t = time.time()
        timestamps.append(df_timestamps.loc[i])
        df_corr = df_log_ret.loc[i-args.delta:i].corr()

        corr_dict = {(index_row, index_col): abs(v) for index_row, row in df_corr.iterrows() for index_col, v in row.items() if index_row != index_col}

        if args.min:
            G = create_min_st(instruments, corr_dict, args.instr_max)
        else:
            G = create_max_st(instruments, corr_dict, args.instr_max)

        paths_list = [l for i1, v in nx.shortest_path_length(G, weight="weight") for l in v.values() if l != 0]
        paths_weighted.append(np.mean(paths_list))

        paths_list = [l for i1, v in nx.shortest_path_length(G) for l in v.values() if l != 0]
        paths_unweighted.append(np.mean(paths_list))

        heterogeneity.append(calc_heterogeneity(G))

        degrees_list.append(dict(G.degree()))
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        leaves_nbr.append(len(degree_sequence) - degree_sequence.index(1))

        if args.plot_graphs:
            plt.figure(figsize=(24,16))
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, node_size=50, with_labels=False)
            plt.savefig("%04d.png"%(i))
            plt.close()

        print("%.2f s"%(time.time() - step_start_t))

    print("Plotting")
    max_degrees_set = {sorted(degrees, key=degrees.get, reverse=True)[0] for degrees in degrees_list}
    max_degrees_dict = {v: [degrees[v] for degrees in degrees_list] for v in max_degrees_set}
    plot_stats_set(timestamps, max_degrees_dict, 'max_degrees.png')

    plot_stats(timestamps, paths_weighted, 'paths weighted', 'paths_weighted.png')
    plot_stats(timestamps, paths_unweighted, 'paths unweighted', 'paths_unweighted.png')
    plot_stats(timestamps, heterogeneity, 'heterogeneity', 'heterogeneity.png')
    plot_stats(timestamps, leaves_nbr, 'leaves number', 'leaves_nbr.png')

