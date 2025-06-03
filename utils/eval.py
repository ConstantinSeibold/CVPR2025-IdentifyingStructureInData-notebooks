# compare clusterings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
import numpy as np

def evaluate_clustering(labels, gt_labels, features, ignore_noise=False):
    if ignore_noise:
        valid_mask = labels != -1  # Ignore noise points (-1) for DBSCAN/HDBSCAN
    else:
        valid_mask = labels
        
    if np.any(valid_mask):
        labels = labels[valid_mask]
        gt_labels = gt_labels[valid_mask]
        features = features[valid_mask]

    ari = adjusted_rand_score(gt_labels, labels)
    nmi = normalized_mutual_info_score(gt_labels, labels)
    fmi = fowlkes_mallows_score(gt_labels, labels)
    silhouette = silhouette_score(features, labels) if len(set(labels)) > 1 else -1  # Silhouette needs >1 cluster

    return {"ARI": ari, "NMI": nmi, "FMI": fmi, "Silhouette": silhouette}
