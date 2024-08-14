import gfkmc
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import random
from tree.gentree import read_tree
import math
import timeit
from typing import List
import itertools
from record_linkage import record_linkage
import os
import sys


def main():
    if len(sys.argv) != 4:
        print("usage: python gfkmc_loop.py [dataset] [k_start] [k_stop]")
        sys.exit(1)

    dataset = sys.argv[1]
    k_start = int(sys.argv[2])
    k_stop = int(sys.argv[3])


    results_dir = f'results/{dataset}/gfkmc'
    os.makedirs(results_dir, exist_ok=True)

    if dataset == 'adult':
        df_orig = pd.read_csv('./datasets/adult_orig.csv')
        df = pd.read_csv('./das/adult_num_anon.csv')[df_orig.columns]
        categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
        numericals = ['age']
        sensitives = ['salary-class']

    elif dataset == 'adult_cat':
        df_orig = pd.read_csv('./datasets/adult_orig.csv').drop('age', axis=1)
        df = pd.read_csv('./das/adult_num_anon.csv')[df_orig.columns]
        categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
        numericals = []
        sensitives = ['salary-class']





    csv_file = f'./{results_dir}/{dataset}_anon_metrics_ncp_{k_start}_{k_stop}.csv'
    with open(csv_file, 'w') as f:
        f.write('k,ncp\n')

    for k in range(k_start, k_stop+1):
        print(f'k={k}')

        att_names = df[categoricals + sensitives].columns
        att_tree = read_tree('./tree/adult/', att_names)
        table = gfkmc.GFKMCTable(df, df_orig, numericals, categoricals, sensitives, att_tree)
        local_start = timeit.default_timer()
        remaining_groups = table.initial_clustering_phase(k)
        stop = timeit.default_timer()
        execution_time = stop - local_start
        print(f"initial_clustering_phase execution time: {execution_time}")

        local_start = timeit.default_timer()
        beta = int(len(remaining_groups) * 0.05)
        table.weighting_phase(beta, remaining_groups)
        stop = timeit.default_timer()
        execution_time = stop - local_start
        print(f"weighting_phase execution time: {execution_time}")

        local_start = timeit.default_timer()
        table.grouping_phase(k)
        stop = timeit.default_timer()
        execution_time = stop - local_start
        print(f"grouping_phase execution time: {execution_time}")

        local_start = timeit.default_timer()
        table.adjustment_phase()
        stop = timeit.default_timer()
        execution_time = stop - local_start
        print(f"adjustment_phase execution time: {execution_time}")

        gen_method = 'cluster_centroid'

        df_anon = table.cluster_generalization(gen_method)

        # df_anon['cluster'] = -1
        # for i, cluster in enumerate(table.clusters):
        #     df_anon.loc[cluster.r_indices, 'cluster'] = i
        #     print(f"cluster={i}, len={cluster.size}")

        ncp_value = table.ncp(df_anon)
        print(f"ncp={ncp_value}")

        # if k % 10 == 0:
        #     matches = record_linkage(df_orig, df_anon, numericals, categoricals, sensitives)
        #     print(matches)
        #     os.makedirs(f'{results_dir}/record_linkage', exist_ok=True)
        #     with open(f'{results_dir}/record_linkage/k{k}.txt', "w", encoding="utf-8") as f:
        #         f.write(f'{matches}')


        df_anon.to_csv(f'./{results_dir}/{dataset}_anon_k{k}.csv', index=False)

        with open(csv_file, 'a') as f:
            # f.write(f'{ncp_value}\n{matches}')
            f.write(f'{k},{ncp_value * 100:.2f}\n')


if __name__ == "__main__":
    main()
