from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
import numpy as np
import itertools
import more_itertools
import math
import matplotlib.pyplot as plt
from itertools import chain, groupby, product,zip_longest
from collections import Counter
from functools import reduce
from itertools import combinations, takewhile
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time
import timeit
from scipy.optimize import linear_sum_assignment



class temp:

    def __init__(self):
      self.x = 1

    def test47(self):
      s = [5,77,9,10, 444]
      t = sorted(range(len(s)), key=lambda k: s[k])
      print(t)
      print([s[i] for i in t[-3:]])

    def test48(self):
      s = [5,77,9,10, 444, 5, 77]
      print(list(more_itertools.locate(s, lambda a: a == np.min(s))))


    def test50(self):
      points = np.array([[0], [3.1], [1], [0.4], [1.5], [0.7], [0.9], [0.5], [30], [31], [30.5]])
      neigh = NearestNeighbors(n_neighbors= 3, radius=1.5, metric='euclidean', n_jobs=-1)
      neigh.fit(points)
      print(neigh.kneighbors(points))
      A = neigh.radius_neighbors_graph(points, mode='distance',sort_results=True)
      print(A.toarray())
      dbscan_instance = DBSCAN(eps=0.4,min_samples = 2,
                               metric='precomputed', algorithm='auto',
                               n_jobs=-1).fit(A)
      print(dbscan_instance.labels_)


    def test(self):
      pass


c = temp()
c.test()





