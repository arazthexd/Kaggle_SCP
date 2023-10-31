import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fclusterdata, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

B_cells_exp = pd.read_csv("../data/temp/B_cells_exp_multiome.csv")
T_cells_CD4_exp = pd.read_csv("../data/temp/T_cells_CD4_exp_multiome.csv")
T_cells_CD8_exp = pd.read_csv("../data/temp/T_cells_CD8_exp_multiome.csv")
Myeloid_exp = pd.read_csv("../data/temp/Myeloid_exp_multiome.csv")
NK_exp = pd.read_csv("../data/temp/NK_exp_multiome.csv")
T_regulatory_exp = pd.read_csv("../data/temp/T_regulatory_exp_multiome.csv")

B_cells_exp_matrix = B_cells_exp.pivot_table("normalized_count", "obs_id", "location")
T_cells_CD4_exp_matrix = T_cells_CD4_exp.pivot_table("normalized_count", "obs_id", "location")
T_cells_CD8_exp_matrix = T_cells_CD8_exp.pivot_table("normalized_count", "obs_id", "location")
Myeloid_exp_matrix = Myeloid_exp.pivot_table("normalized_count", "obs_id", "location")
NK_exp_matrix = NK_exp.pivot_table("normalized_count", "obs_id", "location")
T_regulatory_exp_matrix = T_regulatory_exp.pivot_table("normalized_count", "obs_id", "location")

def mean_fillna(matrix):
    nafilled_matrix = matrix.copy()
    for gene in list(matrix.columns):
        gene_mean = matrix[gene].mean()
        nafilled_matrix[gene].fillna(value=gene_mean, inplace=True)

    return nafilled_matrix


B_cells_filledna = mean_fillna(B_cells_exp_matrix)
T_cells_CD4_filledna = mean_fillna(T_cells_CD4_exp_matrix)
T_cells_CD8_filledna = mean_fillna(T_cells_CD8_exp_matrix)
T_regulatory_filledna = mean_fillna(T_regulatory_exp_matrix)
NK_filledna = mean_fillna(NK_exp_matrix)
Myeloid_filledna = mean_fillna(Myeloid_exp_matrix)

methods = ["single", "complete", "median", "average", "ward", "centroid"]
metrics = ["euclidean"]
def hclust_eval(data, methods, metrics):
    '''
    clusters data using scipy.cluster.hierarchy.linkage 
    evaluates different clustering methods and metrics using scipy.cluster.hierarchhy.cophenet

    parameters:
    data: m*n data matrix with m observations and n variables
    methods: list of methods in the string format
    metrics: list of methods in the string format
    '''

    cophenets = []
    for method in methods:
        for metric in metrics:
            clustering = linkage(data, method=method, metric=metric)
            c, _ = cophenet(clustering, pdist(data, metric=metric))
            cluster_dict = {
                "method":method,
                "metric":metric,
                "c":c,
                "clusters":clustering
            }

        cophenets.append(cluster_dict)

    return cophenets

        