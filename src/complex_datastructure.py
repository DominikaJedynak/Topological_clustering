import gudhi
from gudhi import SimplexTree
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import DisjointSet
import math
from datetime import datetime


class Complex:
    """
    A class for storing max 2D complex build on the set of 3D points.
    """
    
    def __init__(self, points, simplexes=None, max_edge_length=1):
        """
        Constructor creating a complex from either a dictionary of simplexes or as a Rips complex, if dictionary is not
        given.
        :param points: array of 3d points to build the complex on
        :param simplexes: dictionary of simplexes to be included in the complex where number of a point corresponds to
        its position in 'points' array
        :param max_edge_length: the distance used to decide which subsets of points should create a simplex
        """
        self.points = points
        if simplexes is not None:
            self.complex = SimplexTree()
            for d in simplexes:
                for s in simplexes[d]:
                    self.complex.insert(s)
        else:
            self.complex = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length).create_simplex_tree(max_dimension=2)

    def zero_simplexes(self):
        """
        Function returning an array of all zero-simplexes present in the complex.
        """
        return np.array([s[0] for s in self.complex.get_skeleton(0)])
    
    def one_simplexes(self):
        """
        Function returning an array of all one-simplexes present in the complex.
        """
        return np.array([s[0] for s in self.complex.get_skeleton(1) if len(s[0])==2])
    
    def two_simplexes(self):
        """
        Function returning an array of all two-simplexes present in the complex.
        """
        return np.array([s[0] for s in self.complex.get_skeleton(2) if len(s[0])==3])
   
    def connected_components(self):
        """
        Function returning connected components of the complex (disjoint sets of points).
        """
        components = DisjointSet([s for s in self.zero_simplexes().flatten()])
        for (v1, v2) in list(map(tuple, self.one_simplexes())):
            components.merge(v1, v2)
        return components.subsets()

    def count_simplexes(self):
        """
        Function returning the number of zero-,one- and two-simplexes respectively.
        """
        return len(self.zero_simplexes()), len(self.one_simplexes()), len(self.two_simplexes()) 
    
    def print_simplexes(self):
        """
        Function printing all the simplexes present in the complex.
        """
        for simplex in self.complex.get_filtration():
            print("(%s, %.2f)" % tuple(simplex))

    def shortest_path(self, u, v):
        """
        Function returning the shortest path from node u to v consisting only of one-simplexes belonging to the complex.
        :param u: starting point
        :param v: destination point
        """
        if not hasattr(self, 'prev'):
            self.combinatorial_dist()
        path = [v]
        while u != v:
            v = self.prev[u, v]
            path = [v] + path

        return np.array(path)

    def combinatorial_dist(self):
        """
        Function returning a matrix of lengths of the shortest combinatorial paths between all nodes in the complex
        calculated using Floyd-Warshall algorithm TO DO reference. It also creates matrix 'prev' of preceding nodes
        from the paths calculation to allow fast reconstruction of the shortest paths later.
        """
        edges = self.one_simplexes()
        nodes = self.zero_simplexes().flatten()
        N = len(nodes)

        self.dist_matrix = np.full((N, N), math.inf)  # TO DO: possible to optimize?
        self.prev = np.full((N, N), None)
        for u, v in edges:
            self.dist_matrix[u, v] = 1
            self.dist_matrix[v, u] = 1
            self.prev[u][v] = u
            self.prev[v][u] = v
        for u in nodes:
            self.dist_matrix[u, u] = 0
            self.prev[u][u] = u
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if self.dist_matrix[i, j] > self.dist_matrix[i, k] + self.dist_matrix[k, j]:
                        self.dist_matrix[i, j] = self.dist_matrix[i, k] + self.dist_matrix[k, j]
                        self.prev[i][j] = self.prev[k][j]
        return self.dist_matrix
    
    def draw_complex(self, show_now=True, to_file=True):
        """
        Function creating a 3D visualization of the complex using Plotly library.
        :param show_now: flag determining if the complex should be displayed or only a Figure object should be returned
        by the function
        :param to_file: flag determining if the complex should be displayed in a regular way or if an image of it should
        be saved to a file
        """
        triangles = self.two_simplexes()
        lines = self.one_simplexes()

        data = [
            go.Scatter3d(x=self.points[:,0],y=self.points[:,1], z=self.points[:,2], mode='markers'),
        ]

        if len(triangles) > 0:
            data += [go.Mesh3d(
                x = self.points[:,0],
                y = self.points[:,1],
                z = self.points[:,2],
                i = triangles[:,0],
                j = triangles[:,1],
                k = triangles[:,2],
            )]

        for line in lines:
            a = self.points[line[0]]
            b = self.points[line[1]]
            data += [go.Scatter3d(x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]], mode='lines')]

        fig = go.Figure(data=data)
        fig.update_traces(color='lightgrey', selector=dict(type='mesh3d'))
        fig.update_traces(marker_color='lightgrey', selector=dict(type='scatter3d'))
        fig.update_traces(line_width=6, selector=dict(type='scatter3d'))
        fig.update_traces(marker_size=5, selector=dict(type='scatter3d'))
        fig.update_traces(showlegend=False)
        fig.update_layout(autosize=False, width=1000, height=1000)
        if show_now:
            if to_file:
                fig.write_image("complex_" + datetime.now().strftime("%d-%m-%Y_%H-%M") + ".png")
            else:
                fig.show()
        return fig
        
