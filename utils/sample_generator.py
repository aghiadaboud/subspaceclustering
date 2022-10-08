import numpy as np


def generate_samples(n, d, samples_std, seed, subspace_clusters = None):

  np.random.seed(seed)
  samples = np.random.normal(loc=(np.random.uniform(0,0)), scale=samples_std, size=(n, d))
  _95_percent_of_n = int(n*95/100)
  samples[np.ix_(list(range(_95_percent_of_n,n)), list(range(d)))] = np.random.uniform(np.min(samples), np.max(samples), size=(n-_95_percent_of_n,d))

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
          mean.append(np.min(samples[:,dim]) - 2*std)
      else:
        mean.append(np.max(samples[:,dim]) + 2*std)
      dims_to_means.get(dim).append(mean[-1].copy())
      mean_to_std[mean[-1]] = std


    sub_samples = mean + std * np.random.normal(size = (sub_n, sub_d))
    samples[np.ix_(list(cluster_points), list(cluster_features))] = sub_samples
    labels[np.ix_(list(cluster_points), list(cluster_features))] = cluster_label*np.ones((sub_n, sub_d))
    cluster_label = cluster_label + 1

  return (samples, labels)
