de_df = pd.read_parquet(f"{data_dir}/de_train.parquet")
gene_num = len(meta["gene_names"])
mtypess = mtypes.copy()
mtypess.remove("Dimethyl Sulfoxide")
de_array_train = np.zeros((gene_num, len(mtypess), len(ctypes[:4])))
de_array_val = np.zeros((gene_num, len(mtypess), len(ctypes[4:])))
for gene in tqdm(de_df.columns[5:]):
    gene_idx = meta["gene_names"].index(gene)
    gene_pivot_train = de_df.pivot_table(gene, "sm_name", "cell_type").loc[mtypess, ctypes[:4]]
    gene_pivot_val = de_df.pivot_table(gene, "sm_name", "cell_type").loc[mtypess, ctypes[4:]]
    gene_pivot_train = gene_pivot_train.fillna(gene_pivot_train.mean(axis=0))
    de_array_train[gene_idx, :, :] = gene_pivot_train.values
    de_array_val[gene_idx, :, :] = gene_pivot_val.values

de_array_val = de_array_val[:, (np.isnan(de_array_val).sum((0, 2))==0), :][(de_array_train==0).sum((1, 2))==0, :, :]
de_array_train = de_array_train[(de_array_train==0).sum((1, 2))==0, :, :]

with open(f"{data_dir}/de_train.npy", "wb") as f:
    np.save(f, de_array_train)
with open(f"{data_dir}/de_val.npy", "wb") as f:
    np.save(f, de_array_val)