import scipy.cluster.hierarchy as shc
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
import numpy as np
import random
import math

from .complex_datastructure import Complex

class Clustering:
    def __init__(self, my_complex):
        """
        :param my_complex: complex at which trajectories will be located
        """
        self.complex = my_complex
      
    def coefs_to_symbols(self, trajectory):
        """
        :param trajectory: list of 3D points representing the trajectory
        """
        distances = cdist(trajectory, self.complex.points, 'euclidean')  # distances to landmarks
        symbols = np.array([np.argmin(point_to_mesh) for point_to_mesh in distances])      
        return symbols
    
    def symbols_to_coefs(self, trajectory):
        """
        :param trajectory: list of symbols representing the trajectory
        """
        coefs = np.array([self.complex.points[v] for v in trajectory])
        return coefs
    
    def fit(self, trajectories):
        pass # needs to be implemented by children classes
            
    def fit_predict(self, trajectories):
        """
        :param trajectories: list of trajectories (consisting of symbols)
        """
        self.fit(trajectories) 
        return self.clusters
    
    def draw_predict(self, trajectories, show_now=True, on_complex=True):
        """
        :param trajectories: list of trajectories (consisting of symbols)
        :param show_now: flag indicating if the plot should be printed now (or only returned)
        :param on_complex: flag indicating if path should be drawn on a compex or on its own
        """

        if on_complex:
            fig = self.complex.draw_complex(show_now=False)
        else:
            fig = go.Figure()
            fig.update_layout(autosize=False, width=1000, height=1000)
            
        clusters = self.fit_predict(trajectories)
        trajectories_coefs = np.array(list(map(self.symbols_to_coefs, trajectories)))
        color_map = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(max(clusters))]
        
        for path, col in zip(trajectories_coefs, clusters):
            data = []
            data = data + [go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2], mode='markers+lines', line=dict(color="rgb" + str(color_map[col-1]), width=10))]
            fig.add_traces(data)
        fig.update_traces(showlegend=False)

        #print(data)
            
        #fig = go.Figure(data=data)
        
        if show_now:
            fig.show()
        
    

class HierarchicalClustering(Clustering):
             
    def fit(self, trajectories):
        """
        :param trajectories: list of trajectories (consisting of symbols)
        """
        def combinatorial_distance(t1, t2):
            sum = 0
            for p1, p2 in zip(t1,t2):
                sum += dists[int(p1),int(p2)]
            return sum

        dists = self.complex.combinatorial_dist()  # how to pass as an argument
        #trajectories_coefs = np.array(list(map(self.symbols_to_coefs, trajectories)))
        #paths_2d = trajectories_coefs.reshape(trajectories_coefs.shape[0], -1)
        self.clusters = shc.fclusterdata(trajectories, 6, criterion="distance", metric=combinatorial_distance) # TO DO: 8 as parameter! smaller value means less clusters
        return self
    
   
class TopologicalClustering(Clustering):
             
    def fit(self, trajectories):
        """
        :param trajectories: list of trajectories (consisting of symbols)
        """
        iter = 17 #number of iterations to be set
        edges = self.complex.one_simplexes()
        self.clusters = np.ones(len(trajectories), dtype=np.int8)
        for step in range(iter):
            num_clust = max(self.clusters)
            new_cluster = 1
            for cluster in range(1, num_clust+1):
                points_subset = trajectories[list(i for i, c in enumerate(self.clusters) if c == 1), step]
                print(points_subset)
                C = Complex(self.symbols_to_coefs(points_subset), {0: [(p,) for p in points_subset],
                                                    1: [(u,v) if ([u,v] in edges) else print() for u in points_subset for v in points_subset]})
                #C.draw_complex() -  should I renumarate them???
                print(C.connected_components())
                for comp in C.connected_components():
                    for p in comp:
                        for i in range(len(self.clusters)):
                            if trajectories[i,step] == p:
                                self.clusters[i] = new_cluster
                    new_cluster += 1
            print(self.clusters)

                
        return self

    
    
