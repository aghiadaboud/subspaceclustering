import numpy as np
from sklearn.datasets import make_blobs





def generate_sample(n = 0, d = 0, subspace_clusters = None):

  sample = np.random.uniform(0, 1, size=(n,d))
  labels = np.zeros((n, d))
  cluster_label = 1
  for cluster in subspace_clusters:
    cluster_points = cluster[0]
    cluster_features = cluster[1]
    sub_n = cluster[2]
    sub_d = cluster[3]
    std = cluster[4]
    sub_sample, l = make_blobs(n_samples=sub_n, n_features=sub_d, centers=1, cluster_std=std, center_box=(-1, 1), shuffle=True, random_state=None)
    sample[np.ix_(list(cluster_points), list(cluster_features))] = sub_sample
    labels[np.ix_(list(cluster_points), list(cluster_features))] = cluster_label*np.ones((sub_n, sub_d))
    cluster_label = cluster_label + 1

  return (sample, labels)

