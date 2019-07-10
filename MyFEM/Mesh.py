#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 08:56:08 2019

@author: ber0061
"""

import matplotlib.pyplot as plt
import numpy as np


class RectMesh:
    """Generates regular triangular mesh on rectangle."""

    def __init__(self, h_elem=10, w_elem=10, height=1, width=1):
        self.remesh(h_elem, w_elem, height, width)

    def plot(self):
        """ """
        plt.triplot(self.nodes[:, 0], self.nodes[:, 1], self.elems)
        plt.plot(self.nodes[:, 0], self.nodes[:, 1], 'o')

        for j, p in enumerate(self.nodes):
            plt.text(p[0], p[1], j, ha='right',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='None'))  # label the points
        for j, s in enumerate(self.elems):
            p = self.nodes[s].mean(axis=0)
            plt.text(p[0], p[1], '%d' % j, ha='center',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='None'))  # label triangles
        for j, s in enumerate(self.edges):
            p = self.nodes[s].mean(axis=0)
            plt.text(p[0], p[1], '%d' % j, ha='center',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='None'))  # label edges
        plt.show()
        return

    def remesh(self, h_elem=10, w_elem=10, height=1, width=1):
        """
        
        Parameters
        ----------
        h_elem :
            (Default value = 10)
        w_elem :
            (Default value = 10)
        height :
            (Default value = 1)
        width :
            (Default value = 1)

        Returns
        -------

        
        """
        self.h_elem = h_elem
        self.w_elem = w_elem
        self.height = height
        self.width = width

        x1 = np.linspace(0, width, w_elem + 1)
        n2 = h_elem + 1
        x2 = np.linspace(0, height, h_elem + 1)
        [x1, x2] = np.meshgrid(x1, x2)
        self.nodes = np.concatenate((x1.reshape((x1.size, 1), order='F'), x2.reshape((x2.size, 1), order='F')), axis=1)

        m = 2 * w_elem * h_elem
        self.elems = np.zeros((m, 3), dtype='int32')
        idx = 0
        tmp = np.tile(range(0, h_elem), (2, 1))
        temp = tmp.reshape(tmp.size, order='F')
        first = temp.copy()
        second = temp.copy()
        third = temp.copy()
        for i in range(0, w_elem):
            first[:] = temp + i * n2  # vector of first verticles of elements
            second[0:-1] = temp[1:] + (i + 1) * n2  # second verticles
            second[-1] = h_elem + (i + 1) * n2
            third[:] = temp + 1 + i * n2  # third verticles
            third[0::2] = third[0::2] + n2
            # together make elements of column in mesh
            self.elems[idx:(idx + h_elem * 2), 0] = first
            self.elems[idx:(idx + h_elem * 2), 1] = second
            self.elems[idx:(idx + h_elem * 2), 2] = third
            idx += h_elem * 2

        n_vedges = h_elem * (w_elem + 1)
        n_hedges = w_elem * (h_elem + 1)
        n_dedges = h_elem * w_elem
        n_edges = n_vedges + n_hedges + n_dedges
        self.edges = np.zeros((n_edges, 2), dtype='int32')

        # vertical edges goes as points
        idx = 0
        for i in range(0, w_elem + 1):
            self.edges[idx:(idx + h_elem), 0] = range(i * (h_elem + 1), i * (h_elem + 1) + h_elem)
            self.edges[idx:(idx + h_elem), 1] = range(i * (h_elem + 1) + 1, i * (h_elem + 1) + h_elem + 1)
            idx += h_elem

        # horizontal edges
        self.edges[idx:(idx + n_hedges), 0] = range(0, n_hedges)
        self.edges[idx:(idx + n_hedges), 1] = range(0 + (h_elem + 1), n_hedges + (h_elem + 1))
        idx += n_hedges

        # diagonal edges
        for i in range(0, w_elem):
            self.edges[idx:(idx + h_elem), 0] = range(i * (h_elem + 1), i * (h_elem + 1) + h_elem)
            self.edges[idx:(idx + h_elem), 1] = range(i * (h_elem + 1) + 1 + (h_elem + 1),
                                                      i * (h_elem + 1) + h_elem + 1 + (h_elem + 1))
            idx += h_elem

        left = np.array(range(0, h_elem + 1), dtype='int32')
        right = np.array(range((h_elem + 1) * w_elem, (h_elem + 1) * (w_elem + 1)), dtype='int32')
        bottom = np.array(range(0, (h_elem + 1) * (w_elem + 1), h_elem + 1), dtype='int32')
        top = np.array(range(h_elem, (h_elem + 1) * (w_elem + 1), h_elem + 1), dtype='int32')
        self.node_boundary = dict(left=left, right=right, bottom=bottom, top=top)

        left = np.array(range(0, h_elem), dtype='int32')
        right = np.array(range(h_elem * w_elem, h_elem * w_elem + h_elem), dtype='int32')
        bottom = np.array(range(n_vedges, n_vedges + n_hedges, h_elem + 1), dtype='int32')
        top = np.array(range(n_vedges + h_elem, n_vedges + n_hedges, h_elem + 1), dtype='int32')
        self.edge_boundary = dict(left=left, right=right, bottom=bottom, top=top)

        return


# def rect_mesh_delaunay(h_elem=10, w_elem=10, height=1, width=1):
#     x1 = np.linspace(0, width, w_elem + 1)
#     x2 = np.linspace(0, height, h_elem + 1)
#     [x1, x2] = np.meshgrid(x1, x2)
#     node_outer = np.concatenate((x1.reshape((x1.size, 1), order='F'), x2.reshape((x2.size, 1), order='F')), axis=1)
#     x1 = np.linspace(0 + width / w_elem / 2, width - width / w_elem / 2, w_elem)
#     x2 = np.linspace(0 + height / h_elem / 2, height - height / h_elem / 2, h_elem)
#     [x1, x2] = np.meshgrid(x1, x2)
#     node_inter = np.concatenate((x1.reshape((x1.size, 1), order='F'), x2.reshape((x2.size, 1), order='F')), axis=1)
#     node = np.concatenate((node_outer, node_inter), axis=0)
#     tri = Delaunay(node)
#     elem = tri.simplices
#     return [node, elem]
#
#
# def rect_mesh_cross(h_elem=10, w_elem=10, height=1, width=1):
#     x1 = np.linspace(0, width, w_elem + 1)
#     x2 = np.linspace(0, height, h_elem + 1)
#     [x1, x2] = np.meshgrid(x1, x2)
#     node_outer = np.concatenate((x1.reshape((x1.size, 1), order='F'), x2.reshape((x2.size, 1), order='F')), axis=1)
#     x1 = np.linspace(0 + width / w_elem / 2, width - width / w_elem / 2, w_elem)
#     x2 = np.linspace(0 + height / h_elem / 2, height - height / h_elem / 2, h_elem)
#     [x1, x2] = np.meshgrid(x1, x2)
#     node_inter = np.concatenate((x1.reshape((x1.size, 1), order='F'), x2.reshape((x2.size, 1), order='F')), axis=1)
#     node = np.concatenate((node_outer, node_inter), axis=0)
#     temp = range(0, h_elem)
#     temp = np.tile(temp, (4, 1))
#     first_shift = np.array((0, 1, h_elem + 2, h_elem + 1), ndmin=2).T
#     first = temp.copy() + first_shift
#     second_shift = np.array((1, h_elem + 2, h_elem + 1, 0), ndmin=2).T
#     second = temp.copy() + second_shift
#     third = temp.copy() + (w_elem + 1) * (h_elem + 1)
#     m = 4 * w_elem * h_elem
#     elem = np.zeros((m, 3), dtype=int)
#     idx = 0
#     for i in range(0, w_elem):
#         # together make elements of column in mesh
#         elem[idx:(idx + h_elem * 4), 0] = first.reshape(first.size, order='F') + i * (h_elem + 1)
#         elem[idx:(idx + h_elem * 4), 1] = second.reshape(second.size, order='F') + i * (h_elem + 1)
#         elem[idx:(idx + h_elem * 4), 2] = third.reshape(third.size, order='F') + i * h_elem
#         idx = idx + h_elem * 4
#     return [node, elem]
