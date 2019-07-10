#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:35:07 2019

@author: ber0061
"""

import petsc4py
petsc4py.init()
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers


###### TRIANGULAR MESH SETTING ################################################
n=40
my_mesh = Mesh.RectMesh(n, n)

###### PROBLEM SETTING (BOUNDARY + MAT + RHS) #################################
my_problem = ProblemSetting.ProblemInputData(my_mesh) # init problemset. obj
# dirichlet condition setting
dirichlet_boundary = ["right",
                      ["left", [0.5, 1]]] # select boundary
dirichlet_boundary_val = [0,
                          1] # boundary value
my_problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
# neumann condition setting
neumann_boundary = [["top", [0.3, 0.7]],
                    "bottom"] # select boundary
neumann_boundary_val = [-10,
                        0] # boundary value
my_problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val) # set
my_problem.set_rhs(lambda x, y: np.sin(12*x) *np.sin(12*y)) # setting forcing term (rhs)
my_problem.set_material(1) # material setting

###### MATRIX ASSEMBLER (SYSTEM MAT + RHS) ####################################
FEM_assembly = Assemble.LaplaceSteady(my_problem) # init assemble obj
FEM_assembly.assemble_all() # assemble all parts neccesary for solution

print(FEM_assembly.times_assembly)

##### SOLVING using KSP #######################################################
my_solver = Solvers.LaplaceSteady(FEM_assembly) # init

my_solver.ksp_direct_type('umfpack')
#my_solver.ksp_direct_type('mumps')
#my_solver.ksp_direct_type('klu')
#my_solver.ksp_direct_type('cholmod','cholesky')
#my_solver.ksp_direct_type('petsc')
#my_solver.ksp_cg_with_pc('none')
#my_solver.ksp_cg_with_pc('ilu')
#my_solver.ksp_cg_with_pc('jacobi')
#my_solver.ksp_cg_with_pc('sor')
#my_solver.ksp_cg_with_pc('icc')

if n<=30 :
    my_solver.plot_solution() # triplot the solution
else:
    my_solver.plot_solution_image() 

print(my_solver.times_assembly)