#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:18:08 2019

@author: ber0061
"""
import time

import matplotlib.pyplot as plt
from petsc4py import PETSc


class LaplaceSteady:
    """ """

    def __init__(self, assembled_matrices):
        self.assembled_matrices = assembled_matrices
        self.solution = None
        self.ksp = None
        self.times_assembly = dict()

    def init_solution_vec(self):
        """ """
        self.solution = self.assembled_matrices.create_vec()
        return

    def ksp_cg_with_pc(self, pc_type='none'):
        """ Possible preconditioners: ilu, jacobi,sor,... """
        start = time.time()
        if self.solution == None:
            self.init_solution_vec()

        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.assembled_matrices.matrices["A_dirichlet"])
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType(pc_type)
        self.ksp.setFromOptions()

        self.ksp.solve(self.assembled_matrices.rhss["final"], self.solution)
        duration = (time.time() - start)
        self.times_assembly["ksp_cg_" + pc_type] = duration
        return

    def ksp_direct_type(self, solver_type: str = 'umfpack', factor_type: str = 'lu') -> None:
        """ Possible types: umfpack,lu,mumps """
        start = time.time()
        if self.solution == None:
            self.init_solution_vec()

        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.assembled_matrices.matrices["A_dirichlet"])
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        pc.setType(factor_type)
        pc.setFactorSolverType(solver_type)
        pc.setFactorSetUpSolverType()
        self.ksp.setFromOptions()

        self.ksp.solve(self.assembled_matrices.rhss["final"], self.solution)
        duration = (time.time() - start)
        self.times_assembly["ksp_direct_" + solver_type] = duration
        return

    def plot_solution(self):
        """ """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(self.assembled_matrices.problem_setting.geometry.nodes[:, 0],
                        self.assembled_matrices.problem_setting.geometry.nodes[:, 1], self.solution[:],
                        antialiased=True)
        plt.show()
        return

    def plot_solution_image(self):
        """ """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tmp = self.solution[:]
        tmp = tmp.reshape((self.assembled_matrices.problem_setting.geometry.h_elem + 1,
                           self.assembled_matrices.problem_setting.geometry.w_elem + 1), order='F')
        ax.imshow(tmp, extent=[0, 1, 1, 0])
        plt.gca().invert_yaxis()
        plt.show()
        return

    def calculate_window_flow(self):
        """ Calculate flow through boundary Dirichlet windows (using weak form).
        M : petsc4py.PETSc.Mat
            Generalized matrix.
        pressure : petsc4py.PETSc.Vec
            Solution of steady Darcy flow problem.
        dirichlet_boundary : Dict
            Dictionary containing 'idx' with value of list of sub-lists of nodes indexes and 'values' with value of list of
            sub-lists of corresponding values on Dirichlet nodes. 
        """
        M = self.assembled_matrices.matrices['A']
        pressure = self.solution
        dirichlet_boundary = self.assembled_matrices.problem_setting.dirichlet_boundary
        idx = dirichlet_boundary['idx'] # list containing int32 arrays with indices of nodes in Dirichlet windows
        no_windows = len(idx) # int (number of Dirichlet windows)
        self.window_flow = [None] * no_windows # list of scalars (flow through Dirichlet windows)
        t = M * pressure
        multiplicity = 0 * pressure
        for i in range(no_windows):
            multiplicity[idx[i]] += 1
        for i in range(no_windows):
            # correction (preserves zero total flow, allows touching windows)
            tmp = t[idx[i]] / multiplicity[idx[i]]
            self.window_flow[i] = tmp.sum()