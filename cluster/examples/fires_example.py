

from ..fires import fires
from ...utils import read_sample
from ...samples.definitions import SIMPLE_SAMPLES
from ...samples.generator import generate_subspacedata


def test():
  path = SIMPLE_SAMPLES.SAMPLE_SIMPLE1
  sample = read_sample(path)
  fires_instance = fires(sample, 0.4, 2, 3, 2, 2)
  fires_instance.process()
  print(fires_instance.get__pruned_C1())
  print(fires_instance.get_cluster_to_dimension())
  print(fires_instance.get_k_most_similar_clusters())
  print(fires_instance.get__best_merge_candidates())
  print(fires_instance.get__best_merge_clusters())
  print(fires_instance.get__subspace_cluster_approximations())
  clusters = fires_instance.get_clusters()
  for key, value in clusters.items():
   print(key, sorted(value))



def test3():
  sample, lables = generate_subspacedata(50, 10,False, [[10, 4, 1, 1.0], [10, 6, 1, 1.0], [20, 9, 1, 0.6]])
  print(lables)
  data = sample.tolist()
  fires_instance = fires(data, 0.8, 5, 2, 2, 2)
  fires_instance.process()
  print(fires_instance.get__pruned_C1())
  print(fires_instance.get_cluster_to_dimension())
  print(fires_instance.get_k_most_similar_clusters())
  print(fires_instance.get__best_merge_candidates())
  print(fires_instance.get__best_merge_clusters())
  print(fires_instance.get__subspace_cluster_approximations())
  clusters = fires_instance.get_clusters()
  for key, value in clusters.items():
   print(key, value)





test()
