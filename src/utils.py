import numpy as np
from scipy.integrate import RK45
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from plyfile import PlyData


def load_plyfile(filename):
    plydata = PlyData.read(filename)
    coords = np.array([list(p) for p in plydata.elements[0].data])
    simplices = {}
    simplices[0] = [(v,) for v in range(len(coords))]
    simplices[1] = [tuple(e) for [e,_,_,_] in plydata.elements[1].data]
    simplices[2] = [tuple(t) for [t,_,_,_] in plydata.elements[2].data]
    return coords, simplices


def generate_points(system, dimension, starting_point, n_points=1000, step=0.01):
    """
    Function returning an array of 3D points representing a trajectory obtained by solving a given system
    using Runge-Kutta (RK) method.
    :param system: a system of equations used to generate points
    :param dimension: dimension of the equation/of the space to which generated points belong
    :param starting_point: initial state
    :param n_points: length of the trajectory that we want to obtain
    :param step: step size used in a RK method
    """
    integrator = RK45(system, 0, starting_point, 10000, first_step=step, max_step=step)
    points = np.empty((n_points, dimension))

    for _ in range(2000):
        integrator.step()

    for i in range(n_points):
        points[i] = integrator.y
        integrator.step()

    return points


class EpsilonNet:
    """
    Class for creating a mesh consisting of limited number of landmarks approximating given set of points.
    """

    def __init__(self, eps, max_num_of_landmarks=0, method='weighted_furthest'):
        """
        Basic constructor setting all needed parameters.
        :param eps: distance that we want the landmarks to be closer than
        :param max_num_of_landmarks: the maximum number of points that the obtained mesh can consist of
        :param method: the name of the method determining how the distances between points will be validated
        """
        self._eps = eps
        self._max_num_of_landmarks = max_num_of_landmarks if max_num_of_landmarks > 0 else np.iinfo(np.int16).max
        self._method = method
        self._nlandmarks = 0
        self._landmarks = []

    def fit(self, X):
        """
        Function fitting a mesh of points based on distances between given points.
        :param X: points in a [number_of_samples, number_of_features] form which will be used to fit the mesh
        points
        """
        nsamples, nfeatures = X.shape
        self._nlandmarks = 1
        self._landmarks = np.array([X[0]])

        distance_to_landmarks = np.array([np.array(np.linalg.norm(X - self._landmarks[0], axis=1))])
        distance_to_cover = distance_to_landmarks[0]
        while self._nlandmarks < self._max_num_of_landmarks and np.max(distance_to_cover) >= self._eps:
            if self._method == 'furthest_point':
                furthest_point_idx = np.argmax(distance_to_cover)
            else:  # self._method == 'weighted_furthest':
                distance_to_cover = [d if d >= self._eps else 0 for d in distance_to_cover]
                weights = np.power(distance_to_cover / np.max(distance_to_cover), 2)
                weights = weights / np.sum(weights)
                furthest_point_idx = np.random.choice(range(nsamples), p=weights)

            self._landmarks = np.append(self._landmarks, [X[furthest_point_idx]], axis=0)
            distance_to_landmarks = np.append(distance_to_landmarks,
                                              [np.array(np.linalg.norm(X - self._landmarks[self._nlandmarks], axis=1))],
                                              axis=0)
            distance_to_cover = np.min(np.stack((distance_to_cover, distance_to_landmarks[-1])), axis=0)
            self._nlandmarks += 1

        return np.transpose(distance_to_landmarks)

    @property
    def landmarks(self):
        """
        Function returning the set of points that constitute a mesh.
        """
        return self._landmarks


def symbolization(X, lms, eps=0):
    """
    Function used for symbolization of a time series. It transforms a sequence of points represented using
    coordinates into an array of integers (indexes of the elements of the lms cover)
    :param X: an array representing a time series of a shape [number_of_points, dimension]
    :param lms: an array of the shape [_, dimimension] representing landmarks from which we can choose the ones
    being the closest to the points from X
    :param eps: parameter determining maximum allowed distance between real point and its symbolization. If no
    such assignment is possible for some point, symbol '-1' is put instead. Eps equal to 0 means we are just
    looking for the closest landmark
    """
    distances = cdist(X, lms, 'euclidean')
    symbols = np.array([np.argmin(point_to_lms) for point_to_lms in distances])

    if eps > 0:
        symbols = np.array([(l if distances[i, l] < eps else -1) for i, l in enumerate(symbols)])
    return symbols


def kmp_algorithm(pattern, text):
    """
    Function finding all occurrences of a pattern in a text using the Knuth-Morris-Pratt algorithm.
    A list of indices where the pattern is found in the text
    :param pattern: A list representing the pattern to search for
    :param text: The text to search in (list)
    """

    # Compute the failure function of the pattern
    failure = partial_match_kmp(pattern)

    # Search for the pattern in the text using the failure function
    i, j = 0, 0
    matches = []
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == len(pattern):
                matches.append(i - j)
                j = failure[j - 1]
        elif j > 0:
            j = failure[j - 1]
        else:
            i += 1
    return matches


def partial_match_kmp(pattern):
    """
    Function computing the partial match function of a pattern for Knuth-Morris-Pratt algorithm.
    :param pattern: The list representing the pattern to compute the failure function of
    """
    failure = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[j] != pattern[i]:
            j = failure[j - 1]
        if pattern[j] == pattern[i]:
            j += 1
        failure[i] = j
    return failure


@dataclass
class Future:
    """
    Class for storing possible extensions found thanks to the Seer class. It stores a sequence found in
    the history, how many times it occurs and where is starts/ends in the history.
    """
    sequence: tuple
    counter: int
    occurences: list


@dataclass
class Prediction:
    """
    Class for storing predictions obtained thanks to the Seer class. It stores the 'past' that stands for our
    query and a list of possible Futures.
    """
    past: tuple
    futures: list


class Seer:
    """
    Class for making predictions from a time series based on an EpsilonNet.
    """

    def __init__(self, history, lms, eps=0):
        """
        Basic constructor setting all needed parameters.
        :param history: a time series (an array of points represented using coordinates) used to create the
        database for predictions
        :param lms: an array of landmarks defining the set of available symbols
        :param eps: epsilon value used for symbolization
        """
        self._history = history
        self._lms = lms
        self._eps = eps
        self._history_book = symbolization(history, lms)
        self._dimension = len(history[0])
        assert len(lms[0]) == self._dimension

        # state-machine like variables
        self._recent_query = None
        self._recent_story = None
        self._recent_futures = None
        self._recent_prediction = None

    def predict(self, past, f):
        """
        Function searching for possible extensions of a given query of points based on a history provided in the
        constructor.
        :param query: an array of some points represented using coordinates for which we want to find successors
        :param f: function determining how many points we want to predict
        """
        dim = len(past[0])
        assert len(past[0]) == self._dimension

        self._recent_query = past
        self._recent_story = symbolization(self._recent_query, self._lms, self._eps)
        if min(self._recent_story) == -1:
            # the negative symbol means there was no sufficiently good approximation
            print("This sub-trajectory has never happened before")
            return None

        past_instances = kmp_algorithm(self._recent_story, self._history_book)
        futures_dict = dict()
        for idx in past_instances:
            idx_end = idx + len(past) + f
            past_and_future = tuple(self._history_book[idx:idx_end])
            if past_and_future in futures_dict:
                futures_dict[past_and_future].counter += 1
                futures_dict[past_and_future].occurences.append((idx, idx_end))
            else:
                futures_dict[past_and_future] = Future(past_and_future, 1, [(idx, idx_end)])

        self._recent_futures = [f for f in futures_dict.values()]
        self._recent_futures.sort(key=(lambda f: f.counter), reverse=True)

        self._recent_prediction = Prediction(self._recent_story, self._recent_futures)
        return self._recent_prediction

    def paths(self, max_n_paths=100, fix_disconnected=False, complex=None):
        """
        Function returning all paths found by the 'predict' function.
        :param max_n_paths: maximum number of paths that we want to be returned
        :param fix_disconnected: if set to True, we check if all adjacent points from paths create
        one-simplexes in the complex. If not, the path is patched using the shortest path between problematic
        points and both repaired paths and indexes of originaly obtained points are returned.
        :param complex: complex on which paths should be located. Needed when fix_disconnected is set to True
        """
        paths = np.array([future.sequence for future in self._recent_futures])[:max_n_paths]

        if not fix_disconnected:
            return paths

        if complex is None:
            print("Complex should be given if fix_disconnected is set to True!")
            return paths

        edges = complex.one_simplexes().tolist()
        fixed_paths = []
        indices = []
        length = len(paths[0])
        max_length = len(paths[0])
        for path in paths.tolist():  # TO DO get rid of lists here
            idx = [0]
            for i in range(len(path) - 1):
                u = path[idx[i]]
                v = path[idx[i] + 1]
                if u != v and [u, v] not in edges and [v, u] not in edges:
                    connection = complex.shortest_path(u, v).tolist()
                    path = path[:idx[i]] + connection + path[idx[i] + 2:]
                    idx += [idx[-1] + len(connection) - 1]
                    i += len(connection) - 1
                else:
                    idx += [idx[-1] + 1]

            fixed_paths += [path]
            indices += [idx]
            # we need to remember the length of the longest path after patching:
            if idx[-1] + 1 > max_length:
                max_length = idx[-1] + 1

        # to create numpy array from paths we need all of them to be of equal length so we append adequate number
        # of '-1' symbols to shorter paths
        i = 0
        for path in fixed_paths:
            if len(path) < max_length:
                len_dif = max_length - len(path)
                fixed_paths[i] += [-1] * len_dif
            i += 1

        return np.array(fixed_paths), np.array(indices)

