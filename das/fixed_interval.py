import pandas as pd
import numpy as np

import timeit

def fixed_interval(df: pd.DataFrame, QIs: list, k: int):
    start = timeit.default_timer()

    new_df = df.copy()
    ncp = 0
    for QI in QIs:
        # 1: Sort the numerical attributes in ascending order
        new_df = new_df.sort_values(QI)

        # 2: Set threshold θ
        threshold = 1

        # 4: Pick the largest QI Lnum and smallest QI Snum values of QI num
        max_value = new_df[QI].max()
        min_value = new_df[QI].min()

        # 5: Calculate Interval Width (IW)
        iw = (max_value - min_value) // k

        # 6-14: Divide QI num with respect to IW to form equivalence class E and calculate the mean for each E
        bins = list(range(min_value-1, max_value+iw, iw+1))
        print(f'iw={iw}, bins={bins}')

        categories = pd.cut(new_df[QI], bins=bins)
        new_df_categories = pd.DataFrame({'Category': categories})
        new_df = pd.concat([new_df_categories, new_df], axis=1)

        group = new_df.groupby("Category")[QI]
        min_array = np.array(group.min())
        max_array = np.array(group.max())
        count_array = np.array(group.count())

        array = (max_array - min_array) * count_array

        # remove nan values
        array = array[~np.isnan(array)]

        ncp += sum(array / (max_value - min_value))

        # print(new_df)
        group_counts = new_df.groupby('Category')[QI].nunique()
        new_df[QI] = new_df.groupby('Category')[QI].transform(lambda x: x.mean() + threshold if group_counts[x.name] == 1 else x.mean())
        # new_df[QI] = new_df.groupby('Category')[QI].transform(lambda x: x.mean() + threshold if group_counts[x.name] == 1 else x.mean())

        new_df.drop(columns=['Category'], inplace=True)

    new_df.sort_index(inplace=True)

    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')


    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"Execution Time: {execution_time}")

    return new_df
