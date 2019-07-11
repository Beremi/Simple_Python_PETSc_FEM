# Simple_Python_PETSc_FEM
Python-Petsc implementation of some basic FEM discretization (mesh/assembly/selvers) for Darcy flow/elasticity/hydro-mechanic problems/simple fractures

Python package **MyFEM**
Include modules:

###Mesh

- Class **RectUniTri** creates simple uniform triangulation on rectangle. (nodes, elements, edges, dictionaries of boundary nodes/edges (left,right,...))
 
###ProblemSetting

- Class **ProblemInputData** encapsulates Mesh class, contains additional parameters of the problem like boundary conditions, right hand side and material.

###Assembly

- Class **LaplaceSteady** encapsulates ProblemSetting class, contains methods for assembling matrices and right hadn side of the problem

###Solvers

- Class **LaplaceSteady** encapsulates Assembly class, contains different instaces of PETSc ksp solvers
 