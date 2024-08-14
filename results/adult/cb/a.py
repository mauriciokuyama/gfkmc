import pandas as pd

for k in (2,3,4):
    df = pd.read_csv(f'./adult_anon_k{k}.csv', sep=';')
    df.sort_values('ID').drop('ID', axis=1).to_csv(f'./adult_anon_k{k}.csv', index=False)
