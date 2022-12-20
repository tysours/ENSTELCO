#!/usr/bin/env python
import numpy as np
import nlopt
from scipy.optimize import approx_fprime
from ase.units import GPa

from enstelco.strains import STRAIN_SETS
from enstelco.elate.elastic import Elastic


class BaseSolver:
    def __init__(self, strains=None, energies=None):
        self.strains = strains
        self.energies = energies

    @classmethod # XXX: deprecated, not particularly useful
    def start_from_type(cls, strains, energies, lattice_type):
        if lattice_type == 'cubic':
            self = Cubic(strains, energies)
        elif lattice_type == 'hexagonal':
            self = Hexagonal(strains, energies)
        elif lattice_type == 'tetragonal1':
            self = Tetragonal1(strains, energies)
        elif lattice_type == 'tetragonal2':
            self = Tetragonal2(strains, energies)
        elif lattice_type == 'trigonal1':
            self = Trigonal1(strains, energies)
        elif lattice_type == 'trigonal2':
            self = Trigonal2(strains, energies)
        elif lattice_type == 'orthorhombic':
            self = Orthorhombic(strains, energies)
        elif lattice_type == 'monoclinic':
            self = Monoclinic(strains, energies)
        elif lattice_type == 'triclinic':
            self = Triclinic(strains, energies)
        return self

    # TODO: Implement 3rd order elastic consants
    def get_E(self, A2, strains=None):#, A3):
        if strains is None:
            strains = self._strains
        E = 1/2 * A2 * strains**2 #+ 1/6 * A3 * strains**3
        return E

    def _loss(self, x, grad):
        if grad.size > 0:
            grad[:] = approx_fprime(x, self._loss, 0.01 * x, np.array([]))
        #A2, A3 = x
        A2, = x
        guess = self.get_E(A2)#, A3)
        mse = np.mean((guess - self._energies) ** 2)
        return mse

    def fit_energy(self, strains, energies):
        self._strains = strains
        self._energies = energies
        self.opt = nlopt.opt(nlopt.LD_LBFGS, 1)
        self.opt.set_min_objective(self._loss)
        self.opt.set_ftol_rel(1e-6)

        A2 = self.opt.optimize([10.0])
        return A2

    def get_elastic_constants(self):
        self._A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(self._A2) / GPa
        ec_matrix = self.get_ec_matrix()

        elastic_constants = np.linalg.solve(ec_matrix, A2).T[0]
        self.elastic_constants = {o: ec for o, ec in zip(self.order, elastic_constants)}

    def get_elastic_tensor(self):
        if not hasattr(self, 'elastic_constants'):
            self.get_elastic_constants()

        elastic_tensor = np.zeros((6, 6))
        for sym, cij in zip(self.sym, self.order):
            elastic_tensor += sym * self.elastic_constants[cij]

        self.elastic_tensor = elastic_tensor

    def get_properties(self):
        if not hasattr(self, 'elastic_tensor'):
            self.get_elastic_tensor()

        ec = Elastic(self.elastic_tensor.tolist())
        self.properties = ec.averages()

        self.K_V, self.E_V, self.G_V, self.v_V = self.properties[0]
        self.K_R, self.E_R, self.G_R, self.v_R = self.properties[1]
        self.K_VRH, self.E_VRH, self.G_VRH, self.v_VRH = self.properties[2]

        self.K, self.G = self.K_VRH, self.G_VRH # most commonly used

    def get_ec_matrix(self):
        strain_set = STRAIN_SETS[self.lattice_type]
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))
        return ec_matrix


class Cubic(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'cubic'
        self.order = ['C11', 'C44', 'C12']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]])


class Hexagonal(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'hexagonal'
        self.order = ['C11', 'C33', 'C44', 'C12', 'C13']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            ])

    def get_elastic_constants(self):
        self._A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(self._A2) / GPa
        ec_matrix = self.get_ec_matrix()

        elastic_constants = np.linalg.solve(ec_matrix, A2).T[0]
        c11 = elastic_constants[self.order.index('C11')]
        c12 = elastic_constants[self.order.index('C12')]
        c66 = (c11 - c12) / 2
        i_insert = self.order.index('C44') + 1
        self.order.insert(i_insert, 'C66')
        elastic_constants = np.insert(elastic_constants, i_insert, c66)
        c66_sym = np.zeros((6, 6))
        c66_sym[-1, -1] = 1
        self.sym = np.insert(self.sym, i_insert, c66_sym, axis=0)
        self.elastic_constants = {o: ec for o, ec in zip(self.order, elastic_constants)}


class Tetragonal1(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'tetragonal1'
        self.order = ['C11', 'C33', 'C44', 'C66', 'C12', 'C13']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            ])


class Tetragonal2(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'tetragonal2'
        self.order = ['C11', 'C33', 'C44', 'C66', 'C12', 'C13', 'C16']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0,-1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1,-1, 0, 0, 0, 0]],
            ])

class Trigonal1(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'trigonal1'
        self.order = ['C11', 'C33', 'C44', 'C12', 'C13', 'C14']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0],
             [0, 0, 0,-1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1,-1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]],
            ])

    def get_elastic_constants(self):
        self._A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(self._A2) / GPa
        ec_matrix = self.get_ec_matrix()

        elastic_constants = np.linalg.solve(ec_matrix, A2).T[0]
        c11 = elastic_constants[self.order.index('C11')]
        c12 = elastic_constants[self.order.index('C12')]
        c66 = (c11 - c12) / 2
        i_insert = self.order.index('C44') + 1
        self.order.insert(i_insert, 'C66')
        elastic_constants = np.insert(elastic_constants, i_insert, c66)
        c66_sym = np.zeros((6, 6))
        c66_sym[-1, -1] = 1
        self.sym = np.insert(self.sym, i_insert, c66_sym, axis=0)
        self.elastic_constants = {o: ec for o, ec in zip(self.order, elastic_constants)}


class Trigonal2(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        lattice_type = 'trigonal2'
        order = ['C11', 'C33', 'C44', 'C12', 'C13', 'C14', 'C15']
        sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0],
             [0, 0, 0,-1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1,-1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]],
            [[0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0,-1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0,-1],
             [1,-1, 0, 0, 0, 0],
             [0, 0, 0,-1, 0, 0]],
            ])

    def get_elastic_constants(self):
        self._A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(self._A2) / GPa
        ec_matrix = self.get_ec_matrix()

        elastic_constants = np.linalg.solve(ec_matrix, A2).T[0]
        c11 = elastic_constants[self.order.index('C11')]
        c12 = elastic_constants[self.order.index('C12')]
        c66 = (c11 - c12) / 2
        i_insert = self.order.index('C44') + 1
        self.order.insert(i_insert, 'C66')
        elastic_constants = np.insert(elastic_constants, i_insert, c66)
        c66_sym = np.zeros((6, 6))
        c66_sym[-1, -1] = 1
        self.sym = np.insert(self.sym, i_insert, c66_sym, axis=0)
        self.elastic_constants = {o: ec for o, ec in zip(self.order, elastic_constants)}


class Orthorhombic(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'orthorhombic'
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                 'C12', 'C13', 'C23']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            ])


class Monoclinic(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'monoclinic'
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                      'C12', 'C13', 'C23', 'C15', 'C25', 'C35', 'C46']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]],
            ])


class Triclinic(BaseSolver):
    def __init__(self, strains=None, energies=None):
        super().__init__(strains, energies)
        self.lattice_type = 'triclinic'
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                      'C12', 'C13', 'C14', 'C15', 'C16', 'C23', 'C24', 'C25',
                      'C26', 'C34', 'C35', 'C36', 'C45', 'C46', 'C56']
        self.sym = np.array([
            [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]],
            ])


Solvers = {
        'cubic': Cubic,
        'hexagonal': Hexagonal,
        'tetragonal1': Tetragonal1,
        'tetragonal2': Tetragonal2,
        'trigonal1': Trigonal1,
        'trigonal2': Trigonal2,
        'orthorhombic': Orthorhombic,
        'monoclinic': Monoclinic,
        'triclinic': Triclinic,
        }

if __name__ == '__main__':
    #t = Cubic(a)
    energies = np.load('energies.npy')
    strains = np.load('strains.npy')
    #b = t.filter(a)
    t = Base.start(strains, energies, 'triclinic')
