#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 08:51:51 2019

@author: ber0061
"""


from petsc4py import PETSc
import numpy as np
import time



class LaplaceSteady:
    """ """
    problem_setting = None
    matrices = dict()
    rhss = dict()
    times_assembly = dict()
    __elem_areas__ = np.zeros(0)

    def __init__(self, problem_setting):
        self.problem_setting = problem_setting

    def assemble_matrix_generalized(self):
        """ """
        start = time.time()
        elems=self.problem_setting.geometry.elems
        nodes=self.problem_setting.geometry.nodes
        dof = nodes.shape[0]
        m = elems.shape[0]
        S = PETSc.Mat().create()
        S.setSizes([dof,dof])
        S.setType("aij")
        S.setUp()
        S.setPreallocationNNZ(10)

        ve1=np.zeros((m,2),dtype=float)
        ve2=np.zeros((m,2),dtype=float)
        ve3=np.zeros((m,2),dtype=float)
        ve1[:]=nodes[elems[:,2],:]-nodes[elems[:,1],:]
        ve2[:]=nodes[elems[:,0],:]-nodes[elems[:,2],:]
        ve3[:]=nodes[elems[:,1],:]-nodes[elems[:,0],:]
        all_ve=[ve1,ve2,ve3]
        area=0.5*abs(-ve3[:,0]*ve2[:,1]+ve3[:,1]*ve2[:,0])

        sA=np.zeros((m,9),dtype=float)

        for i in range(0,3):
            for j in range(0,3):
                sA[:,i*3+j]=self.problem_setting.material*np.sum(all_ve[i]*all_ve[j],axis=1)/(4*area)

        j=0
        for i in elems:
            S.setValues(i,i,sA[j,:],2)
            j += 1

        S.assemble()
        duration = time.time()-start
        self.matrices["A"]=S
        self.times_assembly["matrices"] = duration
        self.__elem_areas__ = area
        return


    def assemble_rhs_force(self):
        """ """
        start = time.time()
        elems=self.problem_setting.geometry.elems
        dof = self.problem_setting.geometry.nodes.shape[0]
        m = elems.shape[0]
        if self.__elem_areas__.size == 0:
            nodes=self.problem_setting.geometry.nodes
            ve2=np.zeros((m,2),dtype=float)
            ve3=np.zeros((m,2),dtype=float)
            ve2[:]=nodes[elems[:,0],:]-nodes[elems[:,2],:]
            ve3[:]=nodes[elems[:,1],:]-nodes[elems[:,0],:]
            self.__elem_areas__=0.5*abs(-ve3[:,0]*ve2[:,1]+ve3[:,1]*ve2[:,0])

        b = PETSc.Vec().create()
        b.setSizes(dof)
        b.setType("mpi")
        b.set(0)
        b.setUp()

        for i in range(0,3):
            b.setValues(elems[:,i],self.__elem_areas__*self.problem_setting.rhs,2)

        b.assemble()
        b.scale(1/3)
        duration = time.time()-start
        self.rhss["f"]=b
        self.times_assembly["rhsf"] = duration
        return

    def assemble_rhs_neumann(self):
        """ """
        start = time.time()
        edges=self.problem_setting.geometry.edges
        nodes=self.problem_setting.geometry.nodes
        dof = nodes.shape[0]

        b = PETSc.Vec().create()
        b.setSizes(dof)
        b.setType("mpi")
        b.set(0)
        b.setUp()

        j = 0
        for nbsetup in self.problem_setting.neumann_boundary["idx"]:
            indexes = edges[nbsetup,:]
            pointsx = nodes[indexes[:,1],0] - nodes[indexes[:,0],0]
            pointsy = nodes[indexes[:,1],1] - nodes[indexes[:,0],1]
            lengths = np.sqrt(pointsx**2+pointsy**2)
            values = self.problem_setting.neumann_boundary["values"][j]
            b.setValues(indexes[:,0],lengths*values,2)
            b.setValues(indexes[:,1],lengths*values,2)
            j += 1

        b.assemble()
        b.scale(1/2)

        duration = time.time()-start
        self.rhss["N"]=b
        self.times_assembly["rhsN"] = duration
        return


    def assemble_rhs_dirichlet(self):
        """ """
        start = time.time()
        nodes=self.problem_setting.geometry.nodes
        dof = nodes.shape[0]

        b = PETSc.Vec().create()
        b.setSizes(dof)
        b.setType("mpi")
        b.set(0)
        b.setUp()
        matrix = self.matrices["A"]

        j = 0
        for dindexes in self.problem_setting.dirichlet_boundary["idx"]:
            values = self.problem_setting.dirichlet_boundary["values"][j]
            b.setValues(dindexes,values,1)
            j += 1

        b.assemble()
        c = PETSc.Vec().create()
        c.setSizes(dof)
        c.setType("mpi")
        c.set(0)
        c.setUp()
        c.assemble()
        matrix.multAdd(b,c,c)
        c.scale(-1)
        duration = time.time()-start
        self.rhss["Dboundary"]=b
        self.rhss["Dother"]=c
        self.times_assembly["rhsD"] = duration
        return


    def dirichlet_cut_and_sum_rhs(self,duplicate = False):
        """

        Parameters
        ----------
        duplicate :
            (Default value = False)

        Returns
        -------


        """

        start = time.time()
        nodes=self.problem_setting.geometry.nodes
        dof = nodes.shape[0]

        if duplicate:
            matrix = self.matrices["A"].duplicate(copy=True)
            b = PETSc.Vec().create()
            b.setSizes(dof)
            b.setType("mpi")
            b.set(0)
            b.setUp()
            b.assemble()
            b.axpy(alpha=1,x=self.rhss["Dother"])
            b.axpy(alpha=1,x=self.rhss["N"])
            b.axpy(alpha=1,x=self.rhss["f"])
        else:
            matrix = self.matrices["A"]
            b=self.rhss["Dother"]
            b.axpy(alpha=1,x=self.rhss["N"])
            b.axpy(alpha=1,x=self.rhss["f"])

        for dindexes in self.problem_setting.dirichlet_boundary["idx"]:
            matrix.zeroRowsColumns(rows=dindexes,diag=1)
            b.setValues(dindexes,dindexes*0)
        b.assemble()
        b.axpy(alpha=1,x=self.rhss["Dboundary"])

        duration = time.time()-start
        self.rhss["final"]=b
        self.matrices["A_dirichlet"] = matrix
        self.times_assembly["finalize"] = duration
        return

    def cleanup(self):
        """ """
        self.matrices["A"].destroy()
        self.matrices["A_dirichlet"].destroy()
        self.rhss["Dother"].destroy()
        self.rhss["N"].destroy()
        self.rhss["f"].destroy()
        self.rhss["Dboundary"].destroy()
        self.rhss["final"].destroy()
        return

    def assemble_all(self):
        """ """
        self.assemble_matrix_generalized()
        self.assemble_rhs_force()
        self.assemble_rhs_neumann()
        self.assemble_rhs_dirichlet()
        self.dirichlet_cut_and_sum_rhs()
        #self.cleanup()

    def create_vec(self):
        """ """
        nodes=self.problem_setting.geometry.nodes
        dof = nodes.shape[0]
        b = PETSc.Vec().create()
        b.setSizes(dof)
        b.setType("mpi")
        b.set(0)
        b.setUp()
        b.assemble()
        return b

