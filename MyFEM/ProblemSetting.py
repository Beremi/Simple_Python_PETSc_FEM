#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objects encapsulating boundary problem settings. Contains boundary conditions,
right hand sides, materials, ect.

Classes
-------
- `BoundaryValueProblem2D` -- Class encapsulating triangular mesh (from module Mesh), Dirichlet and
    Neumann boundary conditions, material field and right hand side.

"""

import numbers
import numpy as np
from MyFEM import Mesh


class BoundaryValueProblem2D:
    """
    Describes all of the input data needed for the assembly of the finite element system of
    equations. Encapsulates Mesh class instance and adds boundary conditions, materials and right hand sides.

    Parameters
    ----------
    geometry : Mesh module 2D mesh class instance (e.g. RectUniTri)
        Class encapsulating mesh (need attributes: nodes,elems,edges,node_boundary,edge_boundary)

    Attributes
    ----------
    geometry : Mesh module 2D mesh class instance (e.g. RectUniTri)
        Class encapsulating mesh (need attributes: nodes,elems,edges,node_boundary,edge_boundary)
    dirichlet_boundary : Dict
        Dictionary containing 'idx' with value of list of sub-lists of nodes indexes and 'values' with value of list of
        sub-lists of corresponding values on Dirichlet nodes.
    neumann_boundary : Dict
        Dictionary containing 'idx' with value of list of sub-lists of edges indexes and 'values' with value of list of
        sub-lists of corresponding values on Neumann edges.
    rhs : Dict
        Dictionary containing entries corresponding to piece-wise constant functions (numpy arrays of size = number of
        elements), usually corresponding to a right hand side of a PDE.
    material : Dict
        Dictionary containing entries corresponding to piece-wise constant functions (numpy arrays of size = number of
        elements), usually corresponding to a material fields of a PDE.

    Examples
    --------
    >>> problem = BoundaryValueProblem2D(mesh)
    >>> dirichlet_boundary = ["right",
    >>>                       ["left", [0.5, 1]]]  # select boundary and position
    >>> dirichlet_boundary_val = [0,
    >>>                           1]  # boundary value (can be single value, vector of corresponding size or function)
    >>> problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
    >>> # Neumann boundary condition setting
    >>> neumann_boundary = [["top", [0.3, 0.7]],
    >>>                     "bottom"]  # select boundary and position
    >>> neumann_boundary_val = [-10,
    >>>                         0]  # boundary value (can be single value, vector of corresponding size or function)
    >>> problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
    >>> problem.set_rhs(lambda x, y: np.sin(12 * x) * np.sin(12 * y),'f')  # setting forcing term (rhs) (can be single
    >>> # value, vector of corresponding size or function)
    >>> problem.set_material(1, 'permeability')  # set material (can be single value, vector of corresponding size or
    >>> # function)
    """

    def __init__(self, geometry=Mesh.RectUniTri()):
        self.geometry = geometry
        self.dirichlet_boundary = dict()
        self.neumann_boundary = dict()
        self.rhs = dict()
        self.material = dict()

    def set_dirichlet_boundary(self, boundary_mask, boundary_values):
        """

        Parameters
        ----------
        boundary_mask :
            
        boundary_values :
            

        Returns
        -------

        
        """
        d_nodes = []
        d_values = []
        i = 0
        for entry in boundary_values:
            if isinstance(boundary_mask[i], list):
                mask_loc = boundary_mask[i]
                nodes_loc = self.geometry.node_boundary[mask_loc[0]]
                loc_size = nodes_loc.size
                loc_from = int(np.floor(mask_loc[1][0] * loc_size))
                loc_to = int(np.ceil(mask_loc[1][1] * loc_size))
                nodes_loc = nodes_loc[loc_from:loc_to]
            else:
                nodes_loc = self.geometry.node_boundary[boundary_mask[i]]

            if callable(entry):
                values_loc = entry(self.geometry.nodes[nodes_loc, 0], self.geometry.nodes[nodes_loc, 1])
            else:
                if isinstance(entry, numbers.Number):
                    values_loc = np.ones(nodes_loc.size, dtype=float) * entry
                else:
                    values_loc = entry
            d_nodes.append(nodes_loc)
            d_values.append(values_loc)
            self.dirichlet_boundary = dict(idx=d_nodes, values=d_values)
            i += 1
        return

    def set_neumann_boundary(self, boundary_mask, boundary_values):
        """

        Parameters
        ----------
        boundary_mask :
            
        boundary_values :
            

        Returns
        -------

        
        """
        n_edges = []
        n_values = []
        i = 0
        for entry in boundary_values:
            if isinstance(boundary_mask[i], list):
                mask_loc = boundary_mask[i]
                edges_loc = self.geometry.edge_boundary[mask_loc[0]]
                loc_size = edges_loc.size
                loc_from = int(np.floor(mask_loc[1][0] * loc_size))
                loc_to = int(np.ceil(mask_loc[1][1] * loc_size))
                edges_loc = edges_loc[loc_from:loc_to]
            else:
                edges_loc = self.geometry.edge_boundary[boundary_mask[i]]

            if callable(entry):
                midpoints = self.geometry.nodes[self.geometry.edges[edges_loc]].mean(axis=1)
                values_loc = entry(midpoints[:, 0], midpoints[:, 1])
            else:
                if isinstance(entry, numbers.Number):
                    values_loc = np.ones(edges_loc.size, dtype=float) * entry
                else:
                    values_loc = entry
            n_edges.append(edges_loc)
            n_values.append(values_loc)
            self.neumann_boundary = dict(idx=n_edges, values=n_values)
            i += 1
        return

    def set_rhs(self, rhs, desc='f'):
        """

        Parameters
        ----------
        rhs :
        desc :
            

        Returns
        -------

        
        """
        if callable(rhs):
            midpoints = self.geometry.nodes[self.geometry.elems].mean(axis=1)
            values = rhs(midpoints[:, 0], midpoints[:, 1])
        else:
            if isinstance(rhs, numbers.Number):
                values = np.ones(self.geometry.elems.shape[0], dtype=float) * rhs
            else:
                values = rhs

        self.rhs[desc] = values
        return

    def set_material(self, material, desc='permeability'):
        """

        Parameters
        ----------
        material :
        desc :

        Returns
        -------

        
        """
        if callable(material):
            midpoints = self.geometry.nodes[self.geometry.elems].mean(axis=1)
            values = material(midpoints[:, 0], midpoints[:, 1])
        else:
            if isinstance(material, numbers.Number):
                values = np.ones(self.geometry.elems.shape[0], dtype=float) * material
            else:
                values = material

        self.material[desc] = values
        return
