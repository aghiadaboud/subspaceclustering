"""!

@brief Utils that are used by modules of pyclustering.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

"""

import time
import numpy


from numpy import array
from PIL import Image

import matplotlib.pyplot as plt

from sys import platform as _platform

from pyclustering.utils.metric import distance_metric, type_metric


## The number \f$pi\f$ is a mathematical constant, the ratio of a circle's circumference to its diameter.
pi = 3.14159265359


def read_sample(filename, return_type='list'):
    """!
    @brief Returns data sample from simple text file.
    @details This function should be used for text file with following format:
    @code
        point_1_coord_1 point_1_coord_2 ... point_1_coord_n
        point_2_coord_1 point_2_coord_2 ... point_2_coord_n
        ... ...
    @endcode

    As an example there is a 3-dimensional data that contains four points:
    @code
        0.1 0.4 0.1
        0.5 0.6 0.7
        2.3 2.1 2.9
        1.9 2.5 2.0
    @endcode

    In case of this example the following container is going to be produced:
    @code
        [[0.1, 0.4, 0.1], [0.5, 0.6, 0.7], [2.3, 2.1, 2.9], [1.9, 2.5, 2.0]]
    @endcode

    @param[in] filename (string): Path to file with data.
    @param[in] return_type (string): Defines return type of the data (`list` or `numpy`).
    
    @return (array_like) Points where each point represented by coordinates.
    
    """
    
    file = open(filename, 'r')

    if return_type == 'list':
        sample = [[float(val) for val in line.split()] for line in file if len(line.strip()) > 0]
    elif return_type == 'numpy':
        sample = numpy.array([numpy.array([float(val) for val in line.split()])
                              for line in file if len(line.strip()) > 0])
    else:
        raise ValueError("Incorrect 'return_type' is specified '%s'." % return_type)

    file.close()
    return sample


def calculate_distance_matrix(sample, metric=distance_metric(type_metric.EUCLIDEAN)):
    """!
    @brief Calculates distance matrix for data sample (sequence of points) using specified metric (by default Euclidean distance).

    @param[in] sample (array_like): Data points that are used for distance calculation.
    @param[in] metric (distance_metric): Metric that is used for distance calculation between two points.

    @return (list) Matrix distance between data points.

    """

    amount_rows = len(sample)
    return [[metric(sample[i], sample[j]) for j in range(amount_rows)] for i in range(amount_rows)]




def average_neighbor_distance(points, num_neigh):
    """!
    @brief Returns average distance for establish links between specified number of nearest neighbors.
    
    @param[in] points (list): Input data, list of points where each point represented by list.
    @param[in] num_neigh (uint): Number of neighbors that should be used for distance calculation.
    
    @return (double) Average distance for establish links between 'num_neigh' in data set 'points'.
    
    """
    
    if num_neigh > len(points) - 1:
        raise NameError('Impossible to calculate average distance to neighbors '
                        'when number of object is less than number of neighbors.')
    
    dist_matrix = [[0.0 for i in range(len(points))] for _ in range(len(points))]
    for i in range(0, len(points), 1):
        for j in range(i + 1, len(points), 1):
            distance = euclidean_distance(points[i], points[j])
            dist_matrix[i][j] = distance
            dist_matrix[j][i] = distance
            
        dist_matrix[i] = sorted(dist_matrix[i])

    total_distance = 0
    for i in range(0, len(points), 1):
        # start from 0 - first element is distance to itself.
        for j in range(0, num_neigh, 1):
            total_distance += dist_matrix[i][j + 1]
            
    return total_distance / (num_neigh * len(points))


def euclidean_distance(a, b):
    """!
    @brief Calculate Euclidean distance between vector a and b. 
    @details The Euclidean between vectors (points) a and b is calculated by following formula:
    
    \f[
    dist(a, b) = \sqrt{ \sum_{i=0}^{N}(b_{i} - a_{i})^{2}) };
    \f]
    
    Where N is a length of each vector.
    
    @param[in] a (list): The first vector.
    @param[in] b (list): The second vector.
    
    @return (double) Euclidian distance between two vectors.
    
    @note This function for calculation is faster then standard function in ~100 times!
    
    """
    
    distance = euclidean_distance_square(a, b);
    return distance**(0.5);


def euclidean_distance_square(a, b):
    """!
    @brief Calculate square Euclidian distance between vector a and b.
    
    @param[in] a (list): The first vector.
    @param[in] b (list): The second vector.
    
    @return (double) Square Euclidian distance between two vectors.
    
    """  
    
    if ( ((type(a) == float) and (type(b) == float)) or ((type(a) == int) and (type(b) == int)) ):
        return (a - b)**2.0;
    
    distance = 0.0;
    for i in range(0, len(a)):
        distance += (a[i] - b[i])**2.0;
        
    return distance;


def manhattan_distance(a, b):
    """!
    @brief Calculate Manhattan distance between vector a and b.
    
    @param[in] a (list): The first cluster.
    @param[in] b (list): The second cluster.
    
    @return (double) Manhattan distance between two vectors.
    
    """
    
    if ( ((type(a) == float) and (type(b) == float)) or ((type(a) == int) and (type(b) == int)) ):
        return abs(a - b);
    
    distance = 0.0;
    dimension = len(a);
    
    for i in range(0, dimension):
        distance += abs(a[i] - b[i]);
    
    return distance;


def average_inter_cluster_distance(cluster1, cluster2, data = None):
    """!
    @brief Calculates average inter-cluster distance between two clusters.
    @details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
             or by list of indexes of points from the data (represented by list of points), in this case 
             data should be specified.
             
    @param[in] cluster1 (list): The first cluster where each element can represent index from the data or object itself.
    @param[in] cluster2 (list): The second cluster where each element can represent index from the data or object itself.
    @param[in] data (list): If specified than elements of clusters will be used as indexes,
               otherwise elements of cluster will be considered as points.
    
    @return (double) Average inter-cluster distance between two clusters.
    
    """
    
    distance = 0.0;
    
    if (data is None):
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                distance += euclidean_distance_square(cluster1[i], cluster2[j]);
    else:
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                distance += euclidean_distance_square(data[ cluster1[i]], data[ cluster2[j]]);
    
    distance /= float(len(cluster1) * len(cluster2));
    return distance ** 0.5;


def average_intra_cluster_distance(cluster1, cluster2, data=None):
    """!
    @brief Calculates average intra-cluster distance between two clusters.
    @details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
             or by list of indexes of points from the data (represented by list of points), in this case 
             data should be specified.
    
    @param[in] cluster1 (list): The first cluster.
    @param[in] cluster2 (list): The second cluster.
    @param[in] data (list): If specified than elements of clusters will be used as indexes,
               otherwise elements of cluster will be considered as points.
    
    @return (double) Average intra-cluster distance between two clusters.
    
    """
        
    distance = 0.0
    
    for i in range(len(cluster1) + len(cluster2)):
        for j in range(len(cluster1) + len(cluster2)):
            if data is None:
                # the first point
                if i < len(cluster1):
                    first_point = cluster1[i]
                else:
                    first_point = cluster2[i - len(cluster1)]
                
                # the second point
                if j < len(cluster1):
                    second_point = cluster1[j]
                else:
                    second_point = cluster2[j - len(cluster1)]
                
            else:
                # the first point
                if i < len(cluster1):
                    first_point = data[cluster1[i]]
                else:
                    first_point = data[cluster2[i - len(cluster1)]]
            
                if j < len(cluster1):
                    second_point = data[cluster1[j]]
                else:
                    second_point = data[cluster2[j - len(cluster1)]]
            
            distance += euclidean_distance_square(first_point, second_point)
    
    distance /= float((len(cluster1) + len(cluster2)) * (len(cluster1) + len(cluster2) - 1.0))
    return distance ** 0.5


def variance_increase_distance(cluster1, cluster2, data = None):
    """!
    @brief Calculates variance increase distance between two clusters.
    @details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
             or by list of indexes of points from the data (represented by list of points), in this case 
             data should be specified.
    
    @param[in] cluster1 (list): The first cluster.
    @param[in] cluster2 (list): The second cluster.
    @param[in] data (list): If specified than elements of clusters will be used as indexes,
               otherwise elements of cluster will be considered as points.
    
    @return (double) Average variance increase distance between two clusters.
    
    """
    
    # calculate local sum
    if data is None:
        member_cluster1 = [0.0] * len(cluster1[0])
        member_cluster2 = [0.0] * len(cluster2[0])
        
    else:
        member_cluster1 = [0.0] * len(data[0])
        member_cluster2 = [0.0] * len(data[0])
    
    for i in range(len(cluster1)):
        if data is None:
            member_cluster1 = list_math_addition(member_cluster1, cluster1[i])
        else:
            member_cluster1 = list_math_addition(member_cluster1, data[ cluster1[i] ])

    for j in range(len(cluster2)):
        if data is None:
            member_cluster2 = list_math_addition(member_cluster2, cluster2[j])
        else:
            member_cluster2 = list_math_addition(member_cluster2, data[ cluster2[j] ])
    
    member_cluster_general = list_math_addition(member_cluster1, member_cluster2)
    member_cluster_general = list_math_division_number(member_cluster_general, len(cluster1) + len(cluster2))
    
    member_cluster1 = list_math_division_number(member_cluster1, len(cluster1))
    member_cluster2 = list_math_division_number(member_cluster2, len(cluster2))
    
    # calculate global sum
    distance_general = 0.0
    distance_cluster1 = 0.0
    distance_cluster2 = 0.0
    
    for i in range(len(cluster1)):
        if data is None:
            distance_cluster1 += euclidean_distance_square(cluster1[i], member_cluster1)
            distance_general += euclidean_distance_square(cluster1[i], member_cluster_general)
            
        else:
            distance_cluster1 += euclidean_distance_square(data[ cluster1[i]], member_cluster1)
            distance_general += euclidean_distance_square(data[ cluster1[i]], member_cluster_general)
    
    for j in range(len(cluster2)):
        if data is None:
            distance_cluster2 += euclidean_distance_square(cluster2[j], member_cluster2)
            distance_general += euclidean_distance_square(cluster2[j], member_cluster_general)
            
        else:
            distance_cluster2 += euclidean_distance_square(data[ cluster2[j]], member_cluster2)
            distance_general += euclidean_distance_square(data[ cluster2[j]], member_cluster_general)
    
    return distance_general - distance_cluster1 - distance_cluster2


def calculate_ellipse_description(covariance, scale = 2.0):
    """!
    @brief Calculates description of ellipse using covariance matrix.
    
    @param[in] covariance (numpy.array): Covariance matrix for which ellipse area should be calculated.
    @param[in] scale (float): Scale of the ellipse.
    
    @return (float, float, float) Return ellipse description: angle, width, height.
    
    """
    
    eigh_values, eigh_vectors = numpy.linalg.eigh(covariance)
    order = eigh_values.argsort()[::-1]
    
    values, vectors = eigh_values[order], eigh_vectors[order]
    angle = numpy.degrees(numpy.arctan2(*vectors[:,0][::-1]))

    if 0.0 in values:
        return 0, 0, 0

    width, height = 2.0 * scale * numpy.sqrt(values)
    return angle, width, height


def data_corners(data, data_filter = None):
    """!
    @brief Finds maximum and minimum corner in each dimension of the specified data.
    
    @param[in] data (list): List of points that should be analysed.
    @param[in] data_filter (list): List of indexes of the data that should be analysed,
                if it is 'None' then whole 'data' is analysed to obtain corners.
    
    @return (list) Tuple of two points that corresponds to minimum and maximum corner (min_corner, max_corner).
    
    """
    
    dimensions = len(data[0])
    
    bypass = data_filter
    if bypass is None:
        bypass = range(len(data))
    
    maximum_corner = list(data[bypass[0]][:])
    minimum_corner = list(data[bypass[0]][:])
    
    for index_point in bypass:
        for index_dimension in range(dimensions):
            if data[index_point][index_dimension] > maximum_corner[index_dimension]:
                maximum_corner[index_dimension] = data[index_point][index_dimension]
            
            if data[index_point][index_dimension] < minimum_corner[index_dimension]:
                minimum_corner[index_dimension] = data[index_point][index_dimension]
    
    return minimum_corner, maximum_corner


def norm_vector(vector):
    """!
    @brief Calculates norm of an input vector that is known as a vector length.
    
    @param[in] vector (list): The input vector whose length is calculated.
    
    @return (double) vector norm known as vector length.
    
    """
    
    length = 0.0
    for component in vector:
        length += component * component
    
    length = length ** 0.5
    
    return length



def timedcall(executable_function, *args, **kwargs):
    """!
    @brief Executes specified method or function with measuring of execution time.
    
    @param[in] executable_function (pointer): Pointer to a function or method that should be called.
    @param[in] *args: Arguments of the called function or method.
    @param[in] **kwargs:  Arbitrary keyword arguments of the called function or method.
    
    @return (tuple) Execution time and result of execution of function or method (execution_time, result_execution).
    
    """
    
    time_start = time.perf_counter()
    result = executable_function(*args, **kwargs)
    time_end = time.perf_counter()
    
    return time_end - time_start, result

    
    
def draw_clusters(data, clusters, noise = [], marker_descr = '.', hide_axes = False, axes = None, display_result = True):
    """!
    @brief Displays clusters for data in 2D or 3D.
    
    @param[in] data (list): Points that are described by coordinates represented.
    @param[in] clusters (list): Clusters that are represented by lists of indexes where each index corresponds to point in data.
    @param[in] noise (list): Points that are regarded to noise.
    @param[in] marker_descr (string): Marker for displaying points.
    @param[in] hide_axes (bool): If True - axes is not displayed.
    @param[in] axes (ax) Matplotlib axes where clusters should be drawn, if it is not specified (None) then new plot will be created.
    @param[in] display_result (bool): If specified then matplotlib axes will be used for drawing and plot will not be shown.
    
    @return (ax) Matplotlib axes where drawn clusters are presented.
    
    """
    # Get dimension
    dimension = 0;
    if ( (data is not None) and (clusters is not None) ):
        dimension = len(data[0]);
    elif ( (data is None) and (clusters is not None) ):
        dimension = len(clusters[0][0]);
    else:
        raise NameError('Data or clusters should be specified exactly.');
    
    "Draw clusters"
    colors = [ 'red', 'blue', 'darkgreen', 'brown', 'violet', 
               'deepskyblue', 'darkgrey', 'lightsalmon', 'deeppink', 'yellow',
               'black', 'mediumspringgreen', 'orange', 'darkviolet', 'darkblue',
               'silver', 'lime', 'pink', 'gold', 'bisque' ];
               
    if (len(clusters) > len(colors)):
        raise NameError('Impossible to represent clusters due to number of specified colors.');
    
    fig = plt.figure();
    
    if (axes is None):
        # Check for dimensions
        if ((dimension) == 1 or (dimension == 2)):
            axes = fig.add_subplot(111);
        elif (dimension == 3):
            axes = fig.gca(projection='3d');
        else:
            raise NameError('Drawer supports only 2d and 3d data representation');
    
    color_index = 0;
    for cluster in clusters:
        color = colors[color_index];
        for item in cluster:
            if (dimension == 1):
                if (data is None):
                    axes.plot(item[0], 0.0, color = color, marker = marker_descr);
                else:
                    axes.plot(data[item][0], 0.0, color = color, marker = marker_descr);
            
            if (dimension == 2):
                if (data is None):
                    axes.plot(item[0], item[1], color = color, marker = marker_descr);
                else:
                    axes.plot(data[item][0], data[item][1], color = color, marker = marker_descr);
                    
            elif (dimension == 3):
                if (data is None):
                    axes.scatter(item[0], item[1], item[2], c = color, marker = marker_descr);
                else:
                    axes.scatter(data[item][0], data[item][1], data[item][2], c = color, marker = marker_descr);
        
        color_index += 1;
    
    for item in noise:
        if (dimension == 1):
            if (data is None):
                axes.plot(item[0], 0.0, 'w' + marker_descr);
            else:
                axes.plot(data[item][0], 0.0, 'w' + marker_descr);

        if (dimension == 2):
            if (data is None):
                axes.plot(item[0], item[1], 'w' + marker_descr);
            else:
                axes.plot(data[item][0], data[item][1], 'w' + marker_descr);
                
        elif (dimension == 3):
            if (data is None):
                axes.scatter(item[0], item[1], item[2], c = 'w', marker = marker_descr);
            else:
                axes.scatter(data[item][0], data[item][1], data[item][2], c = 'w', marker = marker_descr);
    
    axes.grid(True);
    
    if (hide_axes is True):
        axes.xaxis.set_ticklabels([]);
        axes.yaxis.set_ticklabels([]);
        
        if (dimension == 3):
            axes.zaxis.set_ticklabels([]);
    
    if (display_result is True):
        plt.show();

    return axes;



def set_ax_param(ax, x_title=None, y_title=None, x_lim=None, y_lim=None, x_labels=True, y_labels=True, grid=True):
    """!
    @brief Sets parameters for matplotlib ax.
    
    @param[in] ax (Axes): Axes for which parameters should applied.
    @param[in] x_title (string): Title for Y.
    @param[in] y_title (string): Title for X.
    @param[in] x_lim (double): X limit.
    @param[in] y_lim (double): Y limit.
    @param[in] x_labels (bool): If `True` - shows X labels.
    @param[in] y_labels (bool): If `True` - shows Y labels.
    @param[in] grid (bool): If `True` - shows grid.
    
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    
    if (_platform == "linux") or (_platform == "linux2"):
        rcParams['font.sans-serif'] = ['Liberation Serif']
    else:
        rcParams['font.sans-serif'] = ['Arial']
        
    rcParams['font.size'] = 12

    surface_font = FontProperties()
    if (_platform == "linux") or (_platform == "linux2"):
        surface_font.set_name('Liberation Serif')
    else:
        surface_font.set_name('Arial')
        
    surface_font.set_size('12')
    
    if y_title is not None:
        ax.set_ylabel(y_title, fontproperties = surface_font)
    if x_title is not None:
        ax.set_xlabel(x_title, fontproperties = surface_font)
    
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    if x_labels is False:
        ax.xaxis.set_ticklabels([])
    if y_labels is False:
        ax.yaxis.set_ticklabels([])
    
    ax.grid(grid)


def find_left_element(sorted_data, right, comparator):
    """!
    @brief Returns the element's index at the left side from the right border with the same value as the
            last element in the range `sorted_data`.

    @details The element at the right is considered as target to search. `sorted_data` must
              be sorted collection. The complexity of the algorithm is `O(log(n))`. The
              algorithm is based on the binary search algorithm.

    @param[in] sorted_data: input data to find the element.
    @param[in] right: the index of the right element from that search is started.
    @param[in] comparator: comparison function object which returns `True` if the first argument
                is less than the second.

    @return The element's index at the left side from the right border with the same value as the
             last element in the range `sorted_data`.

    """
    if len(sorted_data) == 0:
        raise ValueError("Input data is empty.")

    left = 0
    middle = (right - left) // 2
    target = sorted_data[right]

    while left < right:
        if comparator(sorted_data[middle], target):
            left = middle + 1
        else:
            right = middle

        offset = (right - left) // 2
        middle = left + offset

    return left


def linear_sum(list_vector):
    """!
    @brief Calculates linear sum of vector that is represented by list, each element can be represented by list - multidimensional elements.
    
    @param[in] list_vector (list): Input vector.
    
    @return (list|double) Linear sum of vector that can be represented by list in case of multidimensional elements.
    
    """
    dimension = 1
    linear_sum = 0.0
    list_representation = (type(list_vector[0]) == list)
    
    if list_representation is True:
        dimension = len(list_vector[0])
        linear_sum = [0] * dimension
        
    for index_element in range(0, len(list_vector)):
        if list_representation is True:
            for index_dimension in range(0, dimension):
                linear_sum[index_dimension] += list_vector[index_element][index_dimension]
        else:
            linear_sum += list_vector[index_element]

    return linear_sum


def square_sum(list_vector):
    """!
    @brief Calculates square sum of vector that is represented by list, each element can be represented by list - multidimensional elements.
    
    @param[in] list_vector (list): Input vector.
    
    @return (double) Square sum of vector.
    
    """
    
    square_sum = 0.0
    list_representation = (type(list_vector[0]) == list)
        
    for index_element in range(0, len(list_vector)):
        if list_representation is True:
            square_sum += sum(list_math_multiplication(list_vector[index_element], list_vector[index_element]))
        else:
            square_sum += list_vector[index_element] * list_vector[index_element]
         
    return square_sum

    
def list_math_subtraction(a, b):
    """!
    @brief Calculates subtraction of two lists.
    @details Each element from list 'a' is subtracted by element from list 'b' accordingly.
    
    @param[in] a (list): List of elements that supports mathematical subtraction.
    @param[in] b (list): List of elements that supports mathematical subtraction.
    
    @return (list) Results of subtraction of two lists.
    
    """
    return [a[i] - b[i] for i in range(len(a))]


def list_math_substraction_number(a, b):
    """!
    @brief Calculates subtraction between list and number.
    @details Each element from list 'a' is subtracted by number 'b'.
    
    @param[in] a (list): List of elements that supports mathematical subtraction.
    @param[in] b (list): Value that supports mathematical subtraction.
    
    @return (list) Results of subtraction between list and number.
    
    """        
    return [a[i] - b for i in range(len(a))]


def list_math_addition(a, b):
    """!
    @brief Addition of two lists.
    @details Each element from list 'a' is added to element from list 'b' accordingly.
    
    @param[in] a (list): List of elements that supports mathematic addition..
    @param[in] b (list): List of elements that supports mathematic addition..
    
    @return (list) Results of addtion of two lists.
    
    """    
    return [a[i] + b[i] for i in range(len(a))]


def list_math_addition_number(a, b):
    """!
    @brief Addition between list and number.
    @details Each element from list 'a' is added to number 'b'.
    
    @param[in] a (list): List of elements that supports mathematic addition.
    @param[in] b (double): Value that supports mathematic addition.
    
    @return (list) Result of addtion of two lists.
    
    """    
    return [a[i] + b for i in range(len(a))]


def list_math_division_number(a, b):
    """!
    @brief Division between list and number.
    @details Each element from list 'a' is divided by number 'b'.
    
    @param[in] a (list): List of elements that supports mathematic division.
    @param[in] b (double): Value that supports mathematic division.
    
    @return (list) Result of division between list and number.
    
    """    
    return [a[i] / b for i in range(len(a))]


def list_math_division(a, b):
    """!
    @brief Division of two lists.
    @details Each element from list 'a' is divided by element from list 'b' accordingly.
    
    @param[in] a (list): List of elements that supports mathematic division.
    @param[in] b (list): List of elements that supports mathematic division.
    
    @return (list) Result of division of two lists.
    
    """    
    return [a[i] / b[i] for i in range(len(a))]


def list_math_multiplication_number(a, b):
    """!
    @brief Multiplication between list and number.
    @details Each element from list 'a' is multiplied by number 'b'.
    
    @param[in] a (list): List of elements that supports mathematic division.
    @param[in] b (double): Number that supports mathematic division.
    
    @return (list) Result of division between list and number.
    
    """    
    return [a[i] * b for i in range(len(a))]


def list_math_multiplication(a, b):
    """!
    @brief Multiplication of two lists.
    @details Each element from list 'a' is multiplied by element from list 'b' accordingly.
    
    @param[in] a (list): List of elements that supports mathematic multiplication.
    @param[in] b (list): List of elements that supports mathematic multiplication.
    
    @return (list) Result of multiplication of elements in two lists.
    
    """        
    return [a[i] * b[i] for i in range(len(a))]
