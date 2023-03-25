import gudhi
import numpy as np
import plotly.graph_objects as go

class Complex:
    """
    A class for storing max 2D complex build on the set of 3D points. Complex can be created from 
    a dict of simplexes on a given set of points or as a Rips complex, if dict is not given 
    """
    
    def __init__(self, points, simplexes=None, max_edge_length=1):
        """
        :param coordinates: sets of 3d points to build the complex on
        :param max_edge_length: the distance used to decide which subsets of points should create a simplex
        """
        self.points = points
        if simplexes is not None:
            self.complex = SimplexTree()
            for d in simplices:
                for s in simplices[d]:
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
        components = DisjointSet(self.zero_simplexes())
        for (v1, v2) in self.one_simplexes():
            components.merge(v1, v2)
        return components.subsets()

    def count_simplexes(self):
        return len(self.zero_simplexes()), len(self.one_simplexes()), len(self.two_simplexes()) 
    
    def list_simplexes(self):
        """
        Function printing all the simplexes present in complex
        """
        for simplex in self.complex.get_filtration():
            print("(%s, %.2f)" % tuple(simplex))       
            
    
    def draw_complex(self, show_now=True):
        triangles = self.two_simplexes()
        lines = self.one_simplexes()

        data = [
            go.Scatter3d(x=self.points[:,0],y=self.points[:,1], z=self.points[:,2], mode='markers'),
        ]

        if len(triangles) > 0:
            data = data + [go.Mesh3d(
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
            data = data + [go.Scatter3d(x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]], mode='lines')]


        fig = go.Figure(data=data)
        fig.update_traces(color='lightgrey', selector=dict(type='mesh3d'))
        fig.update_traces(marker_color='lightgrey', selector=dict(type='scatter3d'))
        fig.update_traces(line_width=6, selector=dict(type='scatter3d'))
        fig.update_traces(marker_size=5, selector=dict(type='scatter3d'))
        fig.update_traces(showlegend=False)
        fig.update_layout(autosize=False, width=1000, height=1000)
        if show_now:
            fig.show()
        return fig
        
