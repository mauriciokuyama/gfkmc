
import gfkmc
import numpy as np
import pandas as pd
import random
from tree.gentree import read_tree
import math
import timeit
import itertools
from record_linkage import record_linkage
import os
import sys


def main():
    if len(sys.argv) != 3:
        print("usage: python record_linkage_loop.py [dataset] [method]")
        sys.exit(1)

    dataset = sys.argv[1]
    method = sys.argv[2]
    k_values = (2, 20, 40, 60, 80, 100)


    results_dir = f'results/{dataset}/{method}'
    os.makedirs(results_dir, exist_ok=True)
    if dataset == 'adult':
        df_orig = pd.read_csv('./datasets/adult_orig.csv')
        categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
        numericals = ['age']
        sensitives = ['salary-class']

    elif dataset == 'adult_cat':
        df_orig = pd.read_csv('./datasets/adult_orig.csv').drop('age', axis=1)
        categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
        numericals = []
        sensitives = ['salary-class']

    # csv_file = f'./{results_dir}/adult_anon_metrics{k_start}_{k_stop}.csv'
    # with open(csv_file, 'w') as f:
    #     f.write('k,ncp\n')

    att_names = df_orig[categoricals].columns
    att_tree = read_tree('./tree/adult/', att_names)

    for k in k_values:
        df_anon = pd.read_csv(f'./{results_dir}/{dataset}_anon_k{k}.csv')

        num_intervals = True
        if method == 'gfkmc' or method == 'gfkmc_most_common_register' or method == 'gfkmc_most_common_values':
            num_intervals = False


        print(f'k={k}')
        matches = record_linkage(df_orig, df_anon, att_tree, numericals, categoricals, num_intervals=num_intervals)
        print(matches)
        os.makedirs(f'{results_dir}/record_linkage', exist_ok=True)
        with open(f'{results_dir}/record_linkage/k{k}.txt', "w", encoding="utf-8") as f:
            f.write(f'{matches}')


        # with open(csv_file, 'a') as f:
        #     # f.write(f'{ncp_value}\n{matches}')
        #     f.write(f'{k},{ncp_value}\n')


if __name__ == "__main__":
    main()
