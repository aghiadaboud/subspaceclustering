import numpy as np
from itertools import chain
from subspaceclustering.cluster.subclu import subclu
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from subspaceclustering.utils.sample_generator import generate_sample
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import csv



def cluster_2d_100n_2sc():
  """samples, labels = generate_sample(100, 2, 1, [[range(40), range(0,1), 0.2],
                                               [range(50, 95), range(1, 2), 0.3]])
  save_data(samples, labels, '2d_100n_2sc')"""

  samples = read_data('2d_100n_2sc')
  subclu_instance = subclu(samples, 0.3, 5)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(samples, clusters, subclu_instance.get_noise())



def cluster_3d_250n_3sc():
  """samples, labels = generate_sample(250, 3, 1, [[tuple(chain(range(50), range(70, 100))), (0,2), 0.3],
                                                [range(120, 190), range(1, 2), 0.3],
                                                [range(200, 235), (0,2), 0.3]])
  save_data(samples, labels, '3d_250n_3sc')"""

  samples = read_data('3d_250n_3sc')
  subclu_instance = subclu(samples, 0.4, 6)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(samples, clusters, subclu_instance.get_noise())




def cluster_4d_500n_4sc():
  """samples, labels = generate_sample(500, 4, 1, [[range(0,160, 2), (0, 3), 0.3], [range(180, 300), (1,2,3), 0.3],
                                                [range(320, 400), (0, 2), 0.4], [range(425, 475), range(2, 3), 0.4]])
  save_data(samples, labels, '4d_500n_4sc')"""

  samples = read_data('4d_500n_4sc')
  subclu_instance = subclu(samples, 0.3, 6)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(samples, clusters, subclu_instance.get_noise())



def cluster_8d_1000n_7sc():
  """samples, labels = generate_sample(1000, 8, 1.5, [[range(250), (1,2,3), 0.25], [range(200, 340), (5,6,7), 0.2],
                                                   [range(290, 390), (1,2), 0.3], [range(380, 510), range(4, 8), 0.25],
                                                   [range(670, 780), range(2, 6), 0.3], [range(800, 900), (0,1), 0.3],
                                                   [range(900, 950), (0,3,4,6,7), 0.15]])
  save_data(samples, labels, '8d_1000n_7sc')"""

  samples = read_data('8d_1000n_7sc')
  subclu_instance = subclu(samples, 0.33, 7)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  visualize_clusters(samples, clusters, subclu_instance.get_noise())



def cluster_16d_2000n_6sc():
  """samples, labels = generate_sample(2000, 16, 2, [[range(150), range(7), 0.2],
                                             [tuple(chain(range(50), range(200, 250))), range(11,16), 0.2],
                                             [range(350,450), range(7,11), 0.2], [range(700, 780), range(9), 0.1],
                                             [range(760, 860), (13,14,15), 0.2], [range(1500,1600), range(2,16), 0.1]])
  save_data(samples, labels, '16d_2000n_6sc')"""

  samples = read_data('16d_2000n_6sc')
  subclu_instance = subclu(samples, 0.5, 8, 'manhattan')
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  for i in [tuple(range(7)), tuple(range(11,16)), tuple(range(7,11)),tuple(range(9)),(13,14,15), tuple(range(2,16))]:
    print(i, list(map(sorted, clusters.get(i, []))), '\n')
  for j in range(16):
    print(j, list(map(sorted, clusters.get(j, []))), '\n')
  #note: clustering result is not correct (eps problem)



def cluster_10d_xn_1sc():
  """for n in range(1000, 10001, 1000):
    samples, labels = generate_sample(n, 10, 2, [[range(500, 600), range(4, 8), 0.15]])
    save_data(samples, labels, 'size_scale/10d_'+str(n)+'n_1sc')"""
  for n in range(1000, 10001, 1000):
    samples = read_data('size_scale/10d_'+str(n)+'n_1sc')
    subclu_instance = subclu(samples, 0.3, 10)
    subclu_instance.process()
    clusters = subclu_instance.get_clusters()
  #note samples 8000-10000 were generated with std 3 instead of 2



def cluster_xd_1000n_1sc():
  """for d in tuple(chain(range(4, 31), (35, 40))):
    samples, labels = generate_sample(1000, d, 1.2, [[range(500, 600), range(1, 4), 0.2]])
    save_data(samples, labels, 'dimensionality_scale/'+str(d)+'d_1000n_1sc')"""
  for d in tuple(chain(range(4, 31), (35, 40))):
    samples = read_data('dimensionality_scale/'+str(d)+'d_1000n_1sc')
    subclu_instance = subclu(samples, 0.3, 9)
    subclu_instance.process()
    clusters = subclu_instance.get_clusters()



def cluster_15d_1000n_1_with_varying_dimensionality_sc():
  """for d in range(1, 15):
    samples, labels = generate_sample(1000, 15, 1.5, [[range(500, 600), range(d), 0.12]])
    save_data(samples, labels, 'dimensionality_of_subspace_cluster_scale/15d_1000n_'+str(d)+'dsc')"""
  for d in range(1, 15):
    samples = read_data('dimensionality_of_subspace_cluster_scale/15d_1000n_'+str(d)+'dsc')
    subclu_instance = subclu(samples, 0.47, 9)
    subclu_instance.process()
    clusters = subclu_instance.get_clusters()



def cluster_15d_1000n_xsc():
  subspaces = [(0,1,2), (4,5,6), (8,9,10), (12,13,14), (0,1,2), (4,5,6), (8,9,10), (12,13,14)]
  for i in range(1, 9):
    """clusters = []
    for j in range(i):
      clusters.append([range(j*100, (j+1)*100), subspaces[j], 0.2])
    samples, labels = generate_sample(1000, 15, 1.5, clusters)
    save_data(samples, labels, 'number_of_subspace_clusters_scale/15d_1000n_'+str(i)+'sc')"""
    samples = read_data('number_of_subspace_clusters_scale/15d_1000n_'+str(i)+'sc')
    subclu_instance = subclu(samples, 0.4, 9)
    subclu_instance.process()
    clusters = subclu_instance.get_clusters()



def read_data(filename):
  samples = []
  with open('subspaceclustering/samples/'+filename+'.csv', newline='') as csvfile:
      reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
      samples = np.array(list(reader)).astype(float)
  return samples


def save_data(samples, labels, filename):
  with open('subspaceclustering/samples/'+filename+'.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in samples:
          writer.writerow(row)

  with open('subspaceclustering/samples/'+filename+'_labels.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in labels:
          writer.writerow(row)



def visualize_clusters(sample, clusters, noise):
  for subspace, list_of_clusters in clusters.items():
    print(subspace, list(map(sorted, list_of_clusters)), '\n')
    """visualizer = cluster_visualizer()
    visualizer.append_clusters(list_of_clusters, sample)
    visualizer.append_cluster(noise.get(subspace, []), sample, marker = 'x')
    visualizer.show()"""

    """visualizer = cluster_visualizer_multidim()
    visualizer.append_clusters(list_of_clusters, sample)
    if noise.get(subspace):
      visualizer.append_cluster(noise.get(subspace), sample, marker = 'x')
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


#cluster_2d_100n_2sc()
#cluster_3d_250n_3sc()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_7sc()
#cluster_16d_2000n_6sc()
#cluster_10d_xn_1sc()
#cluster_xd_1000n_1sc()
#cluster_15d_1000n_1_with_varying_dimensionality_sc()
#cluster_15d_1000n_xsc()
