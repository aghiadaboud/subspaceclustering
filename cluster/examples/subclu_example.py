import numpy as np
from itertools import chain
from subspaceclustering.cluster.subclu import subclu
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from subspaceclustering.utils.sample_generator import generate_sample
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns




def cluster_3d_250n_3sc():
  """sample, labels = generate_sample(250, 3, [[tuple(chain(range(50), range(70, 100))), (0,2), 80, 2, 0.6],
                                            [(range(130, 220)), (range(1, 2)), 90, 1, 0.1],
                                            [range(230, 250), (0,2), 20, 2, 0.3]])
  scale_and_save_data(sample, labels, '3d_250n_3sc')"""

  sample = read_sample("subspaceclustering/samples/3d_250n_3sc.data")
  subclu_instance = subclu(sample, 0.1, 5)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(sample, clusters, subclu_instance.get_noise())
  assert is_present(range(40), clusters.get(0)), 'cluster not found'
  assert is_present(range(40, 140), clusters.get((0, 1))), 'cluster not found'
  assert is_present(range(30), clusters.get((1, 2))), 'cluster not found'


def cluster_4d_500n_4sc():
  """sample, labels = generate_sample(500, 4, [[(range(0,160, 2)), (0, 3), 80, 2, 0.3], [(range(180, 300)), (1,2,3), 120, 3, 0.9],
                                            [(range(350, 400)), (0, 2), 50, 2, 0.6], [(range(450, 500)), range(2, 3), 50, 1, 0.1]])
  scale_and_save_data(sample, labels, '4d_500n_4sc')"""
  #note:last cluster is not found
  sample = read_sample("subspaceclustering/samples/4d_500n_4sc.data")
  subclu_instance = subclu(sample, 0.2, 6)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(sample, clusters, subclu_instance.get_noise())



def cluster_8d_1000n_9sc():
  """sample, labels = generate_sample(1000, 8, [[(range(270)), (1,2,3), 270, 3, 0.6], [(range(50, 150)), (4,5),100, 2, 0.2],
                                             [(range(200, 340)), (5,6,7),140, 3, 0.7], [(range(290, 390)), (1,2),100, 2, 0.3],
                                             [(range(380, 510)), (range(4, 8)),130, 4, 0.9], [(range(520, 640)), (0,1,6),120, 3, 0.5],
                                             [(range(670, 780)), (range(2, 6)),110, 4, 0.6], [(range(880, 950)), (0,1),70, 2, 0.2],
                                             [(range(970, 1000)), (0,3,4,6,7),30, 5, 0.8]])
  scale_and_save_data(sample, labels, '8d_1000n_9sc')"""

  sample = read_sample("subspaceclustering/samples/8d_1000n_9sc.data")
  subclu_instance = subclu(sample, 0.2, 7)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(sample, clusters, subclu_instance.get_noise())


def cluster_16d_2000n_10sc():
  """sample, labels = generate_sample(2000, 16, [[(range(200)), (0,1,2), 200, 3, 0.4], [(range(150)), (4,5,6,7),150, 4, 0.3],
                                             [(range(50)), range(11,16),50, 5, 0.6], [(range(240, 340)), (2,3,4,8,9,10),100, 6, 0.5],
                                             [(range(400, 530)), (0,1,5,6,7,8,9),130, 7, 0.5], [(range(590, 710)), range(7,16),120, 9, 0.5],
                                             [(range(820, 930)), tuple(chain(range(7), range(10, 14))),110, 11, 0.8],
                                             [(range(1080, 1180)), tuple(chain(range(3), range(7, 16))),100, 12, 0.4],
                                             [(range(1490,1600)), range(3,16),110, 13, 0.7],
                                             [(range(1860, 1910)), tuple(chain(range(7), range(9, 16))),50, 14, 0.5]])
  scale_and_save_data(sample, labels, '16d_2000n_10sc')"""

  sample = read_sample("subspaceclustering/samples/16d_2000n_10sc.data")
  subclu_instance = subclu(sample, 0.2, 8)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  print(clusters.get((0,1,2)),'\n')
  print(clusters.get((4,5,6,7)), '\n')
  print(clusters.get((11,12,13,14,15)),'\n')
  print(clusters.get((2,3,4,8,9,10)),'\n')
  print(clusters.get((0,1,5,6,7,8,9)),'\n')
  print(clusters.get((7,8,9,10,11,12,13,14,15)),'\n')
  print(clusters.get(tuple(chain(range(7), range(10, 14)))),'\n')
  print(clusters.get(tuple(chain(range(3), range(7, 16)))),'\n')
  #visualize_clusters(sample, clusters, subclu_instance.get_noise())



def cluster_32d_4000n_15sc():
  sample, labels = generate_sample(4000, 32, [[(range(200)), (0,1,2), 200, 3, 0.4], [(range(150)), (4,5,6,7),150, 4, 0.3],
                                             [(range(50)), range(11,16),50, 5, 0.6], [(range(240, 340)), (2,3,4,8,9,10),100, 6, 0.5],
                                             [(range(400, 530)), (0,1,5,6,7,8,9),130, 7, 0.5], [(range(590, 710)), range(7,16),120, 9, 0.5],
                                             [(range(820, 930)), tuple(chain(range(7), range(10, 14))),110, 11, 0.8],
                                             [(range(1080, 1180)), tuple(chain(range(3), range(7, 16))),100, 12, 0.4],
                                             [(range(1490,1600)), range(3,16),110, 13, 0.7],
                                             [(range(1860, 1910)), tuple(chain(range(7), range(9, 16))),50, 14, 0.5],
                                             [(range(2000, 2100)), range(15,32), 100, 17, 0.4],
                                             [(range(2500, 2600)), tuple(chain(range(4), range(16,32))), 100, 20, 0.6],
                                             [(range(3000, 3100)), tuple(chain(range(7,16), range(14,29))), 100, 24, 0.5],
                                             [(range(3400, 3500)), tuple(chain(range(11), range(16,32))), 100, 27, 0.7],
                                             [(range(3700, 3800)), range(2,32), 100, 30, 0.3]])
  scale_and_save_data(sample, labels, '32d_4000n_15sc')

  sample = read_sample("subspaceclustering/samples/32d_4000n_15sc.data")
  subclu_instance = subclu(sample, 0.3, 8)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()



def cluster_10d_1000n_1sc():
  sample, labels = generate_sample(1000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_1000n_1sc')

def cluster_10d_2000n_1sc():
  sample, labels = generate_sample(2000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_2000n_1sc')

def cluster_10d_3000n_1sc():
  sample, labels = generate_sample(3000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_3000n_1sc')

def cluster_10d_4000n_1sc():
  sample, labels = generate_sample(4000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_4000n_1sc')

def cluster_10d_5000n_1sc():
  sample, labels = generate_sample(5000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_5000n_1sc')

def cluster_10d_6000n_1sc():
  sample, labels = generate_sample(6000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_6000n_1sc')

def cluster_10d_7000n_1sc():
  sample, labels = generate_sample(7000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_7000n_1sc')

def cluster_10d_8000n_1sc():
  sample, labels = generate_sample(8000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_8000n_1sc')

def cluster_10d_9000n_1sc():
  sample, labels = generate_sample(9000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_9000n_1sc')

def cluster_10d_10000n_1sc():
  sample, labels = generate_sample(10000, 10, [[range(500, 600), range(4, 9), 100, 5, 0.3]])
  scale_and_save_data(sample, labels, 'size_scale/10d_10000n_1sc')





def scale_and_save_data(data, labels, name):

  labels_filepath = "subspaceclustering/samples/"+name+"_labels.csv"
  pd.DataFrame(labels).to_csv(labels_filepath)
  scaler = preprocessing.StandardScaler().fit(data)
  sample_scaled = scaler.transform(data)
  data_filepath = "subspaceclustering/samples/"+name+".data"
  file = open(data_filepath, "w+")
  for i in sample_scaled:
    for f in i:
     file.write("%f " % f)
    file.write("\n")
  file.close()



def visualize_clusters(sample, clusters, noise):
  for subspace, list_of_clusters in clusters.items():
    print(subspace, list(map(sorted, list_of_clusters)), '\n')
    """ visualizer = cluster_visualizer()
    visualizer.append_clusters(list_of_clusters, sample)
    visualizer.append_cluster(noise.get(subspace, []), sample, marker = 'x')
    visualizer.show()"""
    """visualizer = cluster_visualizer_multidim()
    visualizer.append_clusters(list_of_clusters, sample)
    visualizer.show()"""

    """df = pd.DataFrame(sample_scaled[np.ix_(list(range(1000)),[0,3,4,6,7])], columns=['col0', 'col3', 'col4', 'col6', 'col7'])
    m = TSNE(learning_rate=50)
    tsne_features = m.fit_transform(df)
    df['x'] = tsne_features[:, 0]
    df['y'] = tsne_features[:, 1]
    sns.scatterplot(x="x", y="y", hue='col0', data=df)
    plt.show()"""


def is_present(cluster, list_of_clusters):
  for item in list_of_clusters:
    if set(cluster) <= set(item):
      return True
  return False


def compute_eps(data, minpts):
  eps_grid = np.linspace(0.3, 1.2, num=10)
  silhouette_scores = []
  for eps in eps_grid:
    model = DBSCAN(eps=eps, min_samples=minpts).fit(np.array(data))
    if len(np.unique(model.labels_)) >1:
      labels = model.labels_
      silhouette_score = round(metrics.silhouette_score(np.array(data), labels), 4)
      silhouette_scores.append(silhouette_score)
      print('eps: ', eps, '--> silhotte: ', silhouette_score)
  return eps_grid[silhouette_scores.index(np.max(silhouette_scores))]



#cluster_3d_250n_3sc()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_9sc()
#cluster_16d_2000n_10sc()
#cluster_32d_4000n_15sc()
