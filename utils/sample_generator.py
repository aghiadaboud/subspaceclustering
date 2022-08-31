import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_sample(n, d, sample_std, subspace_clusters = None):

  sample = np.random.normal(loc=(np.random.uniform(0,0)), scale=sample_std, size=(n, d))
  _95_percent_of_n = int(n*95/100)
  sample[np.ix_(list(range(_95_percent_of_n,n)), list(range(d)))] = np.random.uniform(np.min(sample), np.max(sample), size=(n-_95_percent_of_n,d))

  labels = np.zeros((n, d))
  cluster_label = 1
  dims_to_means = {}
  for j in range(d):
    dims_to_means[j] = []
  mean_to_std = {}
  for cluster in subspace_clusters:
    cluster_points = cluster[0]
    cluster_features = cluster[1]
    sub_n = len(cluster[0])
    sub_d = len(cluster[1])
    std = cluster[2]
    mean = []

    for dim in cluster_features:
      means_of_dim = dims_to_means.get(dim)
      if means_of_dim:
        if len(means_of_dim) % 2 == 0:
          mean.append(3.5*std + 3.5*mean_to_std.get(means_of_dim[-2]) + means_of_dim[-2])
        elif len(means_of_dim) > 1 and len(means_of_dim) % 2 == 1:
          mean.append(-3.5*std + -3.5*mean_to_std.get(means_of_dim[-2]) + means_of_dim[-2])
        else:
          mean.append(np.min(sample[:,dim]) - 2*std)
      else:
        mean.append(np.max(sample[:,dim]) + 2*std)
      dims_to_means.get(dim).append(mean[-1].copy())
      mean_to_std[mean[-1]] = std


    #sub_sample = mean + std * np.random.normal(size = (sub_n, sub_d))
    x, y = make_blobs(n_samples=sub_n, n_features=sub_d, centers=1, cluster_std=std, center_box=(0,0))
    sub_sample = mean + x
    sample[np.ix_(list(cluster_points), list(cluster_features))] = sub_sample
    labels[np.ix_(list(cluster_points), list(cluster_features))] = cluster_label*np.ones((sub_n, sub_d))
    cluster_label = cluster_label + 1
    #plt.scatter(sample[:,0], sample[:,3])
    #plt.show()

  return (sample, labels)
