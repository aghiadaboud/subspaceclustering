from subspaceclustering.cluster.examples import subclu_example
from subspaceclustering.cluster.examples import fires_example
from subspaceclustering.cluster.evaluation import evaluation


# Reproduces Figures 5,6 in the thesis.
subclu_example.plot_D05()
print('\n' * 2)
print('Plotting D05 has finished. The plot is saved in subspaceclustering/images')
print('\n' * 2)

print('******** Creating Table 1 ********')
# Reproduces Table 1 in the thesis.
print('Found clusters by SUBCLU(only in the desired subspaces):')
subclu_example.cluster_D05(print_clusters = True, print_evaluation = False)
# Reproduces Table 1 in the thesis.
print()
print('Found clusters by FIRES(only in the desired subspaces):')
fires_example.cluster_D05(print_clusters = True, print_evaluation = False)
print('\n' * 5)

print('******** Creating Table 2 ********')
# Reproduces Table 2 in the thesis.
print('Found clusters by SUBCLU(only in the desired subspaces):')
subclu_example.cluster_D10(print_clusters = True, print_evaluation = False)
# Reproduces Table 2 in the thesis.
print()
print('Found clusters by FIRES(only in the desired subspaces):')
fires_example.cluster_D10(print_clusters = True, print_evaluation = False)
print('\n' * 5)

# Reproduces Figure 1 in the thesis.
print('******** Creating Figure 1 ********')
fires_example.cluster_8d_1000n_7sc(2,2,2)
print('\n' * 5)
# Reproduces Figure 2 in the thesis.
print('******** Creating Figure 2 ********')
fires_example.cluster_8d_1000n_7sc(2,3,2)
print('\n' * 5)
# Reproduces Figure 3 in the thesis.
print('******** Creating Figure 3 ********')
fires_example.cluster_8d_1000n_7sc(3,4,1)
print('\n' * 5)

# Reproduces Table 3 in the thesis.
print('******** Creating Table 3 ********')
evaluation.compare_clustering_quality()
print('\n' * 5)

# Reproduces Figure 10 in the thesis.
print('******** Creating Figure 10 ********')
evaluation.cluster_10d_xn_2sc()
print('The plot is saved in subspaceclustering/images', '\n' * 3)

# Reproduces Figure 9 in the thesis.
print('******** Creating Figure 9 ********')
evaluation.cluster_10d_xn_10sc()
print('The plot is saved in subspaceclustering/images', '\n' * 3)

# Reproduces Figure 8 in the thesis.
print('******** Creating Figure 8 ********')
evaluation.cluster_xd_1000n_1sc()
print('The plot is saved in subspaceclustering/images', '\n' * 3)

# Reproduces Figure 7 in the thesis.
print('******** Creating Figure 7 ********')
evaluation.cluster_xd_xn_10sc()
print('The plot is saved in subspaceclustering/images')
