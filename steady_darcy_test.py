#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:35:07 2019
Example for Darcy flow.
@author: ber0061
"""

import petsc4py
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers

petsc4py.init()


# TRIANGULAR MESH SETTING -----------------------------------------------------
n = 50
my_mesh = Mesh.RectUniTri(n, n)

# PROBLEM SETTING (BOUNDARY + MAT + RHS) --------------------------------------
my_problem = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
# dirichlet condition setting
dirichlet_boundary = ["right",
                      ["left", [0.5, 1]]]  # select boundary
dirichlet_boundary_val = [0,
                          1]  # boundary value
my_problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
# Neumann boundary condition setting
neumann_boundary = [["top", [0.3, 0.7]],
                    "bottom"]  # select boundary
neumann_boundary_val = [0,
                        0]  # boundary value
my_problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
#my_problem.set_rhs(lambda x, y: np.sin(12 * x) * np.sin(12 * y))  # setting forcing term (rhs)
my_problem.set_rhs(0)
# constant permeability
#my_problem.set_material(1)  # material setting
# permeability as function
def my_material(x,y):
    z=np.sin(10*x)*np.sin(10*y);
    temp=z<0.1
    z[temp]=1
    z[~temp]=10
    return z
my_problem.set_material(my_material)

# MATRIX ASSEMBLER (SYSTEM MAT + RHS) ----------------------------------------
# assemble all parts necessary for solution:
FEM_assembly = Assemble.LaplaceSteady(my_problem)  # init assemble obj
FEM_assembly.assemble_matrix_generalized()
FEM_assembly.assemble_rhs_force()
FEM_assembly.assemble_rhs_neumann()
FEM_assembly.assemble_rhs_dirichlet()
FEM_assembly.dirichlet_cut_and_sum_rhs(duplicate=True)


print(FEM_assembly.times_assembly)

# SOLVING using KSP ----------------------------------------------------------
my_solver = Solvers.LaplaceSteady(FEM_assembly)  # init

#my_solver.ksp_direct_type('umfpack')
#my_solver.ksp_direct_type('mumps')
#my_solver.ksp_direct_type('klu')
#my_solver.ksp_direct_type('cholmod','cholesky')
my_solver.ksp_direct_type('petsc')
#my_solver.ksp_cg_with_pc('none')
#my_solver.ksp_cg_with_pc('ilu')
#my_solver.ksp_cg_with_pc('jacobi')
#my_solver.ksp_cg_with_pc('sor')
#my_solver.ksp_cg_with_pc('icc')

if n <= 30:
    my_solver.plot_solution()  # triplot the solution
else:
    my_solver.plot_solution_image()  # plot solution as image

print(my_solver.times_assembly)

window_flow = my_solver.calculate_boundary_flow()
print(window_flow, sum(window_flow))