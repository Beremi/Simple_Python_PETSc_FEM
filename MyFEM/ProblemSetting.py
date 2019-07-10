#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:06:10 2019

@author: ber0061
"""

import numbers
import numpy as np
from MyFEM import Mesh


class ProblemInputData:
    """ """

    def __init__(self, geometry=Mesh.RectMesh()):
        self.geometry = geometry
        self.dirichlet_boundary = None
        self.neumann_boundary = None
        self.rhs = None
        self.material = None

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

    def set_rhs(self, rhs):
        """

        Parameters
        ----------
        rhs :
            

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

        self.rhs = values
        return

    def set_material(self, material):
        """

        Parameters
        ----------
        material :
            

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

        self.material = values
        return
