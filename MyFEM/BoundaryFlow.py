#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:49:40 2020

@author: simona
"""

class BoundaryFlow:
    """
    Calculate flow through boundary Dirichlet windows
    (using weak form).
    
    Parameters
    ----------
    M : petsc4py.PETSc.Mat
        Generalized matrix.
    pressure : petsc4py.PETSc.Vec
        Solution of steady Darcy flow problem.
    dirichlet_boundary : Dict
        Dictionary containing 'idx' with value of list of sub-lists of nodes indexes and 'values' with value of list of
        sub-lists of corresponding values on Dirichlet nodes.

    Attributes
    ----------
    h_elem : int
        Number of elements in vertical direction.
    w_elem : int
        Number of elements in horizontal direction.
    t : petsc4py.PETSc.Vec
        M * pressure.
    dirichlet_idx : list
        List containing int32 arrays with indices of nodes in Dirichlet windows.
    no_windows : int
        Number of Dirichlet windows.
    window_flow : list
        List of scalars (flow through Dirichlet windows).

    Examples
    --------
    >>> 
    """

    def __init__(self, M, pressure, dirichlet_boundary):
        self.t = M * pressure
        self.dirichlet_idx = dirichlet_boundary['idx']
        self.no_windows = len(self.dirichlet_idx)
        self.window_flow = [None] * self.no_windows
        
    def calculate_boundary_flow(self):
        for i in range(self.no_windows):
            idx = self.dirichlet_idx[i]
            self.window_flow[i] = self.t[idx].sum()