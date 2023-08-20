import scipy.cluster.hierarchy as shc
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import numpy as np
import random
from datetime import datetime
import math
import matplotlib


from .complex_datastructure import Complex


class Clustering:
    """
    A base class for clustering combinatorial trajectories located on a complex.
    """

    def __init__(self, my_complex):
        """
        Basic constructor of the Clustering class.
        :param my_complex: complex on which trajectories will be located
        """
        self.complex = my_complex

    def coefs_to_symbols(self, trajectory):
        """
        Function transforming a trajectory represented as an array of 3D points into a symbolic representation by
        assigning to each point from the trajectory the index of the closest point (zero-simplex) from the complex.
        :param trajectory: an array of 3D points representing the trajectory
        """
        distances = cdist(trajectory, self.complex.points, 'euclidean')
        symbols = np.array([np.argmin(point_to_mesh) for point_to_mesh in distances])
        return symbols

    def symbols_to_coefs(self, trajectory):
        """
        Function transforming a trajectory represented in a symbolic way into an array of 3D points by assigning to each
        point from the trajectory its coordinates.
        :param trajectory: an array of symbols representing the trajectory
        """
        coefs = np.array([self.complex.points[v] for v in trajectory if v != -1])
        return coefs

    def fit_predict(self, trajectories, *params):
        """
        Function delegating clustering of the trajectories to the 'fit' function and returning obtained clusters.
        :param trajectories: an array of trajectories (consisting of symbols)
        :param params: parameters needed for clustering - dependent on the child class
        """
        if len(params) == 2:
            self.fit(trajectories, params[0], params[1])
        elif len(params) == 1:
            self.fit(trajectories, params[0])
        else:
            self.fit(trajectories)
        return self.clusters

    def draw_predict(self, trajectories, *params, on_complex=True, to_file=True):
        """
        Function creating a 3D visualization of the clustered trajectories using Plotly library.
        :param trajectories: an array of trajectories (consisting of symbols)
        :param on_complex: flag indicating if clustered paths should be drawn on a complex or on its own
        :param to_file: flag determining if the figure should be displayed in a regular way or if an image of it should
        be saved to a file
        """

        if on_complex:
            fig = self.complex.draw_complex(show_now=False)
        else:
            fig = go.Figure()
            fig.update_layout(autosize=False, width=1000, height=1000)

        if len(params) == 2:
            clusters = self.fit_predict(trajectories, params[0], params[1])
        elif len(params) == 1:
            clusters = self.fit_predict(trajectories, params[0])
        else:
            clusters = self.fit_predict(trajectories)

        if isinstance(trajectories, tuple):
            trajectories = trajectories[0]

        trajectories_coefs = list(map(self.symbols_to_coefs, trajectories))
        color_palette = matplotlib.cm.get_cmap('jet')

        for path, col in zip(trajectories_coefs, clusters):
            data = []
            color = "rgb" + str(tuple(int(x*256) for x in color_palette(col/max(clusters))[:-1]))
            data = data + [go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='markers+lines',
                                        line=dict(color=color, width=10))]
            fig.add_traces(data)
        fig.update_traces(showlegend=False)
        camera = dict(
            eye=dict(x=-1.5, y=1.0, z=0.2)
        )
        fig.update_layout(scene_camera=camera)

        if to_file:
            params = str([str(p) + "_" for p in params])
            fig.write_image(self.__class__.__name__ + "_" + params + datetime.now().strftime(
                "%d-%m-%Y_%H-%M") + ".png")
        else:
            fig.show()

    def plot_clusters_transitions(self, trajectories, param_values):
        """
        Function plotting sankey diagram which represents how new clusters emerged and vanished as the parameter's value
        was changing.
        """
        results = []
        m = 1
        last_clusters = [0] * len(trajectories)
        used_params = []

        # printing outcomes of clustering for increasing parameter values:
        for p in param_values:
            if isinstance(p,tuple):
                self.fit_predict(trajectories, p[0], p[1])
            else:
                self.fit_predict(trajectories, p)
            print(self.clusters)
            if (self.clusters != last_clusters).any():
                results += [self.clusters]
                used_params += [p]
                last_clusters = self.clusters.copy()
            if max(self.clusters) > m:
                m = max(self.clusters)
        n = len(self.clusters)
        steps = len(results)

        source_nodes = []
        target_nodes = []
        values = []
        color_palette = matplotlib.cm.get_cmap('jet')

        source_nodes_val = [x + 1 for l in [[i] * m for i in range(m)] for x in l]
        target_nodes_val = [x + 1 for l in [list(range(m)) for i in range(m)] for x in l]

        # for every jump from i-th parameter value to (i+1)th:
        for s in range(steps - 1):

            # table storing numbers of transitions
            val_temp = [0] * (len(source_nodes_val))

            for i, (sn, tn) in enumerate(zip(source_nodes_val, target_nodes_val)):
                for j in range(n):
                    # we check how many trajectories change their cluster from sn to tn in that step
                    if results[s][j] == sn and results[s + 1][j] == tn:
                        val_temp[i] += 1

            # every time we need to increase cluster's indexes as we want flow to be left to right, not circular
            source_nodes += [s * m + x for l in [[i] * m for i in range(m)] for x in l]
            target_nodes += [(s + 1) * m + x for l in [list(range(m)) for i in range(m)] for x in l]
            values += val_temp

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["c" + str(i % m + 1) + " for p=" + "{:.2f}".format(used_params[i // m][1]) for i in range(m * steps)],
                color=["rgb" + str(tuple(int(x * 256) for x in color_palette((col+1) / max(results[i]))[:-1])) for i in range(steps) for col in
                       range(m)]
            ),
            link=dict(
                source=source_nodes,
                target=target_nodes,
                value=values
            ))])
        fig.show()


class HierarchicalClustering(Clustering):
    """
    A base class for clustering combinatorial trajectories on a complex using Scipy hierarchical clustering.
    """

    def fit(self, trajectories, metric, t):
        """
        Function doing the clustering of given trajectories.
        :param trajectories: an array of trajectories (consisting of symbols)
        :param metric: metric used for determining the distances between trajectories needed for clustering
        :param t: scalar used as threshold when forming clusters required by 'fclusterdata' function
        """
        self.clusters = shc.fclusterdata(trajectories, t, criterion="distance", metric=metric)
        return self


class CombinatorialHierarchicalClustering(HierarchicalClustering):
    """
    Class for clustering combinatorial trajectories using combinatorial distance between points on complex.
    """

    def fit(self, trajectories, t):
        """
        Function doing the clustering of given trajectories using combinatorial distance which sums up how many edges
        (one-simplexes of the complex) it requires to get from i-th vertex in one trajectory to corresponding i-th
        vertex in another trajectory.
        :param trajectories: an array of trajectories (consisting of symbols) which should be of equal length
        :param t: scalar used as threshold when forming clusters required by 'fclusterdata' function
        """
        def combinatorial_distance(t1, t2):
            assert len(t1) == len(t2) # or allow?
            if hasattr(self.complex, "dist_matrix"):
                dists = self.complex.dist_matrix
            else:
                dists = self.complex.combinatorial_dist()
            sum = 0
            for p1, p2 in zip(t1, t2):
                sum += dists[int(p1), int(p2)]
            return sum/len(t1)

        return super().fit(trajectories, combinatorial_distance, t)


class DTWHierarchicalClustering(HierarchicalClustering):
    """
    Class for clustering combinatorial trajectories using Dynamic Time Warping (DTW) between points on complex.
    """

    def fit(self, trajectories, t):
        """
        Function doing the clustering of given trajectories using DTW.
        :param trajectories: an array of trajectories (consisting of symbols) not necessarily of equal length
        :param t: scalar used as threshold when forming clusters required by 'fclusterdata' function
        """
        def DTW_distance(t1, t2):
            if hasattr(self.complex, "dist_matrix"):
                dists = self.complex.dist_matrix
            else:
                dists = self.complex.combinatorial_dist()
            DTW = np.full((len(t1)+1, len(t2)+1), math.inf)
            DTW[0, 0] = 0
            for i in range(1, len(t1)+1):
                for j in range(1, len(t2)+1):
                    d = dists[int(t1[i-1]), int(t2[j-1])]
                    DTW[i, j] = d + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])

            return 2 * DTW[-1, -1] / (len(t1) + len(t2))

        return super().fit(trajectories, DTW_distance, t)


class ConnectedComponentsClustering(Clustering):
    """
    Class for clustering combinatorial trajectories by analyzing connected components of sub-complexes spanned by the
    points belonging to trajectories.
    """

    def fit(self, trajectories, iter, window=1):
        """
        Function doing the clustering of given trajectories by analyzing connected components of sub-complexes.
        :param trajectories: an array of trajectories (consisting of symbols)
        :param iter: integer value defining number of steps of the algorithm
        :param window: integer value defining how many points of each trajectory we consider in each step of the
        algorithm
        """
        patched_paths = False
        if isinstance(trajectories, tuple):  # which means that we have trajectories of different lengths and original
                                             # indexes were passed along with points
            original_idx = trajectories[1]
            trajectories = trajectories[0]
            patched_paths = True
            length = len(original_idx[0])
        else:
            length = len(trajectories[0])

        num_traj = len(trajectories)
        edges = self.complex.one_simplexes().tolist()
        self.clusters = np.ones(num_traj, dtype=np.int8)
        new_clusters = np.ones(num_traj, dtype=np.int8)

        for step in range(iter):
            if step + window > length:
                print("Paths too short for that request!")
                break
            # we check how many clusters have been created up to now
            num_clust = max(self.clusters)
            new_cluster = 1
            for cluster in range(1, num_clust + 1):
                # we create sub-complexes spanned by the points from trajectories of indexes 'step' to 'step+window'
                if not patched_paths:
                    # the subset of points from step to step+window of all trajectories in cluster 'cluster'
                    points_subset = np.unique(trajectories[list(i for i, c in enumerate(self.clusters) if c == cluster),
                                              step:step+window].flatten())
                else:
                    # we want to consider closer half of the points added between original points during patching
                    points_subset = []
                    begin = original_idx[:, step] - (original_idx[:, step] - original_idx[:, max(step-1, 0)]) // 2
                    end = original_idx[:, step+window] + (original_idx[:, min(step+window+1, len(original_idx[0])-1)] -
                                                          original_idx[:, step+window]) // 2
                    trajectories_in_cluster = list(i for i, c in enumerate(self.clusters) if c == cluster)
                    for t in trajectories_in_cluster:
                        points_subset += list(trajectories[t, begin[t]:end[t]].flatten())
                    points_subset = np.unique(np.array(points_subset))

                spanned_edges = []
                spanned_points = []
                for u in points_subset:
                    #if u not in spanned_points:
                    spanned_points += [(u,)]
                    for v in points_subset:
                        if ([min(u,v), max(u,v)] in edges) and ((min(u,v), max(u,v)) not in spanned_edges):
                            spanned_edges += [(min(u,v), max(u,v))]
                C = Complex(self.symbols_to_coefs(points_subset), {0: spanned_points,
                                                                   1: spanned_edges})
                # trajectories whose points belong to one connected component are put in the same cluster
                for comp in C.connected_components():
                    for p in comp:
                        flag = False
                        for i in range(num_traj):
                            # if it was in cluster that we analyze and belongs to current component:
                            if self.clusters[i] == cluster and trajectories[i, step] == p:
                                new_clusters[i] = new_cluster
                                flag = True
                        #if not flag:
                        #    print("Panic")
                    if len(comp) > 0 and flag:
                        new_cluster += 1
                    #print(new_clusters)
            self.clusters = np.copy(new_clusters)
        return self


class HodgeLaplacianClustering(Clustering):
    """
    Class for clustering combinatorial trajectories TO DO.
    """

    def fit(self, trajectories, eps=1, min_s=1):
        # we follow the notation from paper TO DO link
        nr_points, nr_edges, nr_triangles = self.complex.count_simplexes()
        B1 = np.zeros((nr_points, nr_edges))
        B2 = np.zeros((nr_edges, nr_triangles))

        points = self.complex.zero_simplexes().tolist()
        edges = self.complex.one_simplexes().tolist() # we need to convert it to list to be able to use 'index' method
        for i in range(nr_edges):
            u, v = edges[i]
            B1[points.index([u]), i] = -1
            B1[points.index([v]), i] = 1

        triangles = self.complex.two_simplexes()
        for i in range(nr_triangles):
            u, v, w = triangles[i]
            B2[edges.index([u, v]), i] = 1
            B2[edges.index([v, w]), i] = 1
            B2[edges.index([u, w]), i] = -1

        L1 = np.matmul(np.transpose(B1), B1) + np.matmul(B2, np.transpose(B2))
        eigen_val, eigen_vec = np.linalg.eigh(L1)

        U_harm = eigen_vec[:, abs(eigen_val - 0) < 0.1 ** 10]
        if len(U_harm) == 0:
            print("No zero eigenvalues therefore the embedding is 0-dimensional! Exiting.")
            return

        f = np.zeros((nr_edges, len(trajectories)))
        for i in range(len(trajectories)):
            t = trajectories[i].tolist()
            for j in range(len(t)-1):
                u = t[j]
                v = t[j+1]
                if v == -1:
                    break
                if u < v:
                    f[edges.index([u, v]), i] = 1
                elif u > v:
                    f[edges.index([v, u]), i] = -1

        f_emb = np.matmul(np.transpose(U_harm), f)

        self.clusters = DBSCAN(eps=eps, min_samples=min_s).fit(np.transpose(f_emb)).labels_ + np.array([1])

        return f_emb #self



