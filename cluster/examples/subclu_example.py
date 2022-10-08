import numpy as np
from itertools import chain
from subspaceclustering.cluster.subclu import subclu
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from subspaceclustering.utils.sample_generator import generate_samples
from subspaceclustering.cluster.clusterquality import quality_measures
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import umap
import umap.plot
import time




def cluster_vowel():
  #528 x 11
  samples = read_csv('real_world_data/vowel')
  labels = samples[:,-1]
  true_clusters = { tuple(range(10)) : [] }
  for label in np.setdiff1d(labels, -1):
    true_clusters.get(tuple(range(10))).append(np.where(labels == label)[0])
  subclu_instance = subclu(samples[:, :-1], 1, 9)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(528, 10, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())

def cluster_diabetes():
  #768 x 9
  samples = read_csv('real_world_data/diabetes')
  labels = samples[:,-1]
  true_clusters = { (0,1,2,3,4,5,6,7) : [] }
  for label in np.setdiff1d(labels, -1):
    true_clusters.get((0,1,2,3,4,5,6,7)).append(np.where(labels == label)[0])
  subclu_instance = subclu(samples[:, :-1], 24, 15)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(768, 8, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())

def cluster_glass():
  #214 x 10
  samples = read_csv('real_world_data/glass')
  labels = samples[:,-1]
  true_clusters = { tuple(range(9)) : [] }
  for label in np.setdiff1d(labels, -1):
    true_clusters.get(tuple(range(9))).append(np.where(labels == label)[0])
  subclu_instance = subclu(samples[:, :-1], 0.89, 4)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(214, 9, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())


def cluster_D05():
  true_clusters = {
    (0,1,4) : [range(0, 302)],
    (0,1,3,4) : [range(0, 156), range(1008,1158), range(1159, 1309)],
    (1,2,3) : [range(303, 604)],
    (1,2,3,4) : [range(303, 459)],
    (0,2,3) : [range(605, 755), range(1310, 1460)],
    (1,2,4) : [tuple(chain(range(605,655), range(756, 854)))],
    (0,2,3,4) : [range(855, 1007)]
    }
  samples = read_csv('db_dimensionality_scale/D05')
  samples = np.delete(samples, -1, 1)

  pca = PCA(n_components=3)
  samples = pca.fit_transform(samples)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(samples[:,0], samples[:,1], samples[:,2])
  plt.show()

  subclu_instance = subclu(samples, 25,25)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(1595, 5, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())


def cluster_D10():
  true_clusters = {
    (1,2,3,5,9) : [range(0, 300)],
    (0,1,2,3,5,6,7,9) : [range(0, 150)],
    (0,2,4,6,9) : [range(301, 602)],
    (0,1,2,3,4,5,6,9) : [range(301, 454)],
    (1,2,4,5,7,8) : [range(603, 753)],
    (0,2,3,4,5,9) : [tuple(chain(range(603,653), range(754, 853)))],
    (1,2,3,4,5,6,7,9) : [range(854, 1003)],
    (0,2,3,4,5,6,8,9) : [range(1004, 1154)],
    (0,1,2,3,6,7,8) : [range(1155, 1305)],
    (0,3,4,6,7,8) : [range(1306,1457)]
    }
  samples = read_csv('db_dimensionality_scale/D10')
  samples = np.delete(samples, -1, 1)
  subclu_instance = subclu(samples, 25,20)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  #evaluate_clustering(1595, 10, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())



def cluster_2d_100n_2sc():
  true_clusters = {}
  true_clusters[0] = [range(40)]
  true_clusters[1] = [range(50, 95)]
  samples, labels = generate_samples(100, 2, 1, 0, [[range(40), range(0,1), 0.2],
                                                   [range(50, 95), range(1, 2), 0.3]])
  subclu_instance = subclu(samples, 0.3, 5)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(100, 2, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())



def cluster_3d_250n_3sc():
  true_clusters = {
    (0,2) : [tuple(chain(range(50), range(70, 100))), range(200, 235)],
    1 : [range(120, 190)]
    }
  samples, labels = generate_samples(250, 3, 1, 1, [[tuple(chain(range(50), range(70, 100))), (0,2), 0.3],
                                                    [range(120, 190), range(1, 2), 0.3],
                                                    [range(200, 235), (0,2), 0.3]])
  subclu_instance = subclu(samples, 0.4, 6)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(250, 3, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())



def cluster_4d_500n_4sc():
  true_clusters = {
    (0,3) : [range(0,160, 2)],
    (1,2,3) : [range(180, 300)],
    (0,2) : [range(320, 400)],
    2 : [range(425, 475)]
    }
  samples, labels = generate_samples(500, 4, 1, 0, [[range(0,160, 2), (0, 3), 0.3], [range(180, 300), (1,2,3), 0.3],
                                                    [range(320, 400), (0, 2), 0.4], [range(425, 475), range(2, 3), 0.4]])
  subclu_instance = subclu(samples, 0.3, 6)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(500, 4, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())



def cluster_8d_1000n_7sc():
  true_clusters = {
    (1,2,3) : [range(250)],
    (5,6,7) : [range(200, 340)],
    (1,2) : [range(290, 390)],
    (4,5,6,7) : [range(380, 510)],
    (2,3,4,5) : [range(670, 780)],
    (0,1) : [range(800, 900)],
    (0,3,4,6,7) : [range(900, 950)]
    }
  samples, labels = generate_samples(1000, 8, 1.5, 2, [[range(250), (1,2,3), 0.25], [range(200, 340), (5,6,7), 0.25],
                                                       [range(290, 390), (1,2), 0.3], [range(380, 510), range(4, 8), 0.25],
                                                       [range(670, 780), range(2, 6), 0.3], [range(800, 900), (0,1), 0.3],
                                                       [range(900, 950), (0,3,4,6,7), 0.3]])
  subclu_instance = subclu(samples, 0.6, 7)
  subclu_instance.process()
  found_clusters = subclu_instance.get_clusters()
  evaluate_clustering(1000, 8, true_clusters, found_clusters)
  print_clustering(found_clusters, subclu_instance.get_noise())



def cluster_10d_xn_2sc():
  runtime = []
  size = list(range(1000, 10001, 1000))
  for n in size:
    samples, labels = generate_samples(n, 10, 3, 1, [[range(400, 550), range(4, 8), 0.2],
                                                     [range(700, 850), range(7), 0.15]])
    subclu_instance = subclu(samples, 0.37, 10)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    runtime.append(stop - start)
    #print(f"in {(stop - start):0.4f} seconds",'\n')
    #print(f"in {(toc - tic)*1000:0.4f} ms")
  plt.figure(figsize=(8, 4))
  plt.plot(size, runtime, label='SUBCLU')
  plt.xlabel("Number of points")
  plt.ylabel("Runtime in s")
  plt.legend(loc='upper left')
  plt.savefig('cluster_10d_xn_2sc.pdf')
  plt.show()


def cluster_10d_xn_10sc():
  #We will consider only the first 10 dimensions.
  runtime = []
  datasets = [1500, 2500, 3500, 4500, 5500]
  for dataset in datasets:
    samples = read_csv('db_size_scale/S'+ str(dataset))
    samples = samples[:, :10] / 100
    subclu_instance = subclu(samples, 0.7, 10)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    runtime.append(stop - start)
  plt.figure(figsize=(8, 4))
  plt.plot(datasets, runtime, label='SUBCLU')
  plt.xlabel("Number of points")
  plt.ylabel("Runtime in s")
  plt.legend(loc='upper left')
  plt.savefig('cluster_10d_xn_10sc.pdf')
  plt.show()



def cluster_xd_1000n_1sc():
  runtime = []
  dimensionalities = [10, 15, 20, 25, 30, 35]
  for d in dimensionalities:
    samples, labels = generate_samples(1000, d, 1.5, 1, [[range(700, 850), range(d-7, d), 0.2]])
    subclu_instance = subclu(samples, 0.49, 10)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    runtime.append(stop - start)
  plt.figure(figsize=(8, 4))
  plt.plot(dimensionalities, runtime, label='SUBCLU')
  plt.xlabel("Data dimensionality")
  plt.ylabel("Runtime in s")
  plt.legend(loc='upper left')
  plt.savefig('cluster_xd_1000n_1sc.pdf')
  plt.show()


def cluster_xd_1000n_7sc():
  #We will consider only the first 1000 points.
  runtime = []
  datasets = ['05', '10', '15', '20']
  for dataset in datasets:
    samples = read_csv('db_dimensionality_scale/D'+dataset)
    samples = samples[:1000, :-1] / 100
    subclu_instance = subclu(samples, 0.3, 10)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    runtime.append(stop - start)
  plt.figure(figsize=(8, 4))
  plt.plot(list(map(int, datasets)), runtime,label='SUBCLU')
  plt.xlabel("Data dimensionality")
  plt.ylabel("Runtime in s")
  plt.legend(loc='upper left')
  plt.savefig('cluster_xd_1000n_7sc.pdf')
  plt.show()



def evaluate_clustering(db_size, db_dimensionality, true_clusters, found_clusters):
  print('F1 Recall %.2f' % quality_measures.overall_f1_recall(true_clusters, found_clusters))
  print('F1 Merge %.2f' % quality_measures.overall_f1_merge(true_clusters, found_clusters))
  print('RNIA %.2f' % quality_measures.overall_rnia(db_size, db_dimensionality, true_clusters, found_clusters))
  print('CE %.2f' % quality_measures.overall_ce(db_size, db_dimensionality, true_clusters, found_clusters))
  print('E4SC %.2f' % quality_measures.e4sc(true_clusters, found_clusters))


def read_csv(filename):
  samples = None
  with open('subspaceclustering/samples/'+filename+'.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    samples = np.array(list(reader)).astype(float)
  return samples


def save_csv(samples, labels, filename):
  with open('subspaceclustering/samples/'+filename+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in samples:
        writer.writerow(row)

  with open('subspaceclustering/samples/'+filename+'_labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in labels:
        writer.writerow(row)



def print_clustering(clusters, noise):
  for subspace, list_of_clusters in clusters.items():
    print(subspace, list(map(sorted, list_of_clusters)), '\n')


def visualize_samples(samples):
  mapper = umap.UMAP().fit(samples.data)
  umap.plot.points(mapper)
  plt.show()

  """pca = PCA(n_components=3)
  samples = pca.fit_transform(samples)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  colours = ['green', 'b', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'tomato']
  for indicies in true_clusters.values():
    for cluster in indicies:
      ax.scatter(samples[cluster,0], samples[cluster,1], samples[cluster,2], c = colours.pop())
  plt.show()"""


def visualize_clusters():

  """visualizer = cluster_visualizer_multidim()
  visualizer.append_clusters(list_of_clusters, list(samples))
  if len(noise.get(subspace)) > 0:
    visualizer.append_cluster(noise.get(subspace), list(samples), marker = 'x')
  visualizer.show()"""



#cluster_vowel()
#cluster_diabetes()
#cluster_glass()
#cluster_D05()
#cluster_D10()
#cluster_D15()
#cluster_2d_100n_2sc()
#cluster_3d_250n_3sc()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_7sc()
#cluster_10d_xn_2sc()
#cluster_10d_xn_10sc()
#cluster_xd_1000n_1sc()
#cluster_xd_1000n_7sc()
#cluster_15d_1000n_1_with_varying_dimensionality_sc()
