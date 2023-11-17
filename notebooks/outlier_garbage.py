import dask.dataframe as dd
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math
from scipy.cluster.hierarchy import fclusterdata, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
import seaborn as sns

import sys
sys.setrecursionlimit(10000)

print("loading file...")
B_cells_filledna = pd.read_csv("../data/temp/B_cells_normalized_filledna.csv")
B_cells_filledna.set_index("obs_id", inplace=True)
data_scaler = StandardScaler()
B_cells_scaled_data = data_scaler.fit_transform(B_cells_filledna)
print("starting linkage...")
B_cells_average_euclidean_clustering = linkage(B_cells_scaled_data, method="single", metric="euclidean")

print("starting figure")
fig = plt.figure(figsize=(300,10), dpi=120)
ax = fig.add_axes(rect=[1,1,1,1])

print("starting dendrogram...")
dendrogram(B_cells_average_euclidean_clustering, ax=ax)
print("saving figure...")
fig.savefig("../data/temp/B_cells_single_dendrogram.png", dpi=120)