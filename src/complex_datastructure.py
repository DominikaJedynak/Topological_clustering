import gudhi
from gudhi import SimplexTree
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import DisjointSet
import math
from datetime import datetime

class Complex:
    """
    A class for storing max 2D complex build on the set of 3D points. Complex can be created from 
    a dict of simplexes on a given set of points or as a Rips complex, if dict is not given 
    """
    
    def __init__(self, points, simplexes=None, max_edge_length=1):
        """
        :param points: sets of 3d points to build the complex on
        :param simplexes:
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
        return np.array([s[0] for s in self.complex.get_skeleton(0)])
    
    def one_simplexes(self):
        return np.array([s[0] for s in self.complex.get_skeleton(1) if len(s[0])==2])
    
    def two_simplexes(self):
        return np.array([s[0] for s in self.complex.get_skeleton(2) if len(s[0])==3])
   
    def connected_components(self):
        components = DisjointSet([s for s in self.zero_simplexes().flatten()])
        for (v1, v2) in list(map(tuple, self.one_simplexes())):
            components.merge(v1, v2)
        return components.subsets()

    def count_simplexes(self):
        return len(self.zero_simplexes()), len(self.one_simplexes()), len(self.two_simplexes()) 
    
    def list_simplexes(self):
        """
        Function printing all the simplexes present in the complex
        """
        for simplex in self.complex.get_filtration():
            print("(%s, %.2f)" % tuple(simplex))

    #floyd_warshall algorithm
    def combinatorial_dist(self):
        """
        Returns a matrix of lengths of the shortest paths between all nodes
        """
        edges = self.one_simplexes()
        nodes = self.zero_simplexes().flatten()
        N = len(nodes)

        dist_matrix = np.full((N, N), math.inf)  # only upper half of this matrix will be really used
        for u, v in edges:
            dist_matrix[u, v] = 1
            dist_matrix[v, u] = 1
        for u in nodes:
            dist_matrix[u, u] = 0
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        return dist_matrix
    
    def draw_complex(self, show_now=True, to_file=True):
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
        # show now needed temporarily because of how ipynb operates
        if show_now:
            if to_file:
                fig.write_image("complex_" + datetime.now().strftime("%d-%m-%Y_%H-%M") + ".png")
            else:
                fig.show()
        return fig
        
