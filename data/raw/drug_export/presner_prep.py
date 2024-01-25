import pandas as pd


df = pd.read_csv('data/gp_scripts.txt', sep='\t')
df = df[["drug_name", "quantity"]].drop_duplicates()
df = df.dropna(subset=["drug_name"])  # otherwise presner raises error at the end of the run
df.to_csv('data/presner_input.txt', sep='\t', index=False)