import scipy.cluster.hierarchy as shc
import plotly.graph_objects as go
import numpy as np
from complex_datastructure import Complex


class HierarchicalClustering:
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
        
    def fit(self, trajectories):
        """
        :param trajectories: list of trajectories (consisting of 3D points)
        """
        paths_2d = trajectories.reshape(trajectories.shape[0], -1)  
        clusters = shc.fclusterdata(paths_2d, 5, criterion="distance") # TO DO: 5 as parameter!
        return self
        
    def fit_predict(self, trajectories):
        """
        :param trajectories: list of trajectories (consisting of 3D points)
        """
        paths_2d = trajectories.reshape(trajectories.shape[0], -1)  
        clusters = shc.fclusterdata(paths_2d, 8, criterion="distance")
        return clusters
    
    def draw_predict(self, trajectories, show_now=True, on_complex=True):
        """
        :param trajectories: list of trajectories (consisting of 3D points)
        :param show_now: flag indicating if the plot should be printed now (or only returned)
        :param on_complex: flag indicating if path should be drawn on a compex or on its own
        """
        def color_dict(nr):
            if nr==1:
                return 'green'
            if nr==2:
                return 'orange'
            if nr==3:
                return 'magenta'
            if nr==4:
                return 'blue'
        
        if on_complex:
            fig = self.complex.draw_complex(show_now=False)
        else:
            fig = go.Figure()
            fig.update_layout(autosize=False, width=1000, height=1000)
            
        clusters = self.fit_predict(trajectories)
        
        for path, col in zip(trajectories, clusters):
            #print(path)
            #print(path[:0])
            #fig.data[].line.color = "#ffe476"
            data = []
            data = data + [go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2], mode='markers+lines', line=dict(color=color_dict(col), width=10))]
            fig.add_traces(data)
        fig.update_traces(showlegend=False)

        #print(data)
            
        #fig = go.Figure(data=data)
        
        if show_now:
            fig.show()
        
