#!/usr/bin/env python
import numpy as np
import nlopt
from scipy.optimize import approx_fprime
from ase.units import GPa
from enstelco.strains import STRAIN_SETS
import elastic_coudert


class Base:
    def __init__(self, strains, energies):
        self.strains = strains
        self.energies = energies


    # XXX: WTF DO I NAME THIS?
    @classmethod
    def start(cls, strains, energies, lattice_type):
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
    def get_E(self, A2):#, A3):
        E = 1/2 * A2 * self._strains**2 #+ 1/6 * A3 * self._strains**3
        return E


    def loss(self, x, grad):
        if grad.size > 0:
            grad[:] = approx_fprime(x, self.loss, 0.01 * x, np.array([]))
        #A2, A3 = x
        A2, = x
        guess = self.get_E(A2)#, A3)
        mse = np.mean((guess - self._energies) ** 2)
        return mse


    def fit_energy(self, strains, energies):
        self._strains = strains
        self._energies = energies
        self.opt = nlopt.opt(nlopt.LD_LBFGS, 1)
        self.opt.set_min_objective(self.loss)
        self.opt.set_ftol_rel(1e-6)
        
        A2 = self.opt.optimize([10.0])
        return A2


    def get_elastic_constants(self):
        A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(A2) / GPa
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

        ec = elastic_coudert.Elastic(self.elastic_tensor.tolist())
        self.properties = ec.averages()
        self.K, self.G = self.properties[2][0], self.properties[2][2]
        

### XXX: Probably should rework this, not a huge need for subclasses
class Cubic(Base):

    def get_ec_matrix(self):
        self.order = ['C11', 'C44', 'C12']
        strain_set = STRAIN_SETS['cubic']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
 
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

class Hexagonal(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C33', 'C44', 'C12', 'C13']
        strain_set = STRAIN_SETS['hexagonal']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

    def get_elastic_constants(self):
        A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(A2) / GPa
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

class Tetragonal1(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C33', 'C44', 'C66', 'C12', 'C13']
        strain_set = STRAIN_SETS['tetragonal1']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

class Tetragonal2(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C33', 'C44', 'C66', 'C12', 'C13', 'C16']
        strain_set = STRAIN_SETS['tetragonal2']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix


class Trigonal1(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C33', 'C44', 'C12', 'C13', 'C14']
        strain_set = STRAIN_SETS['trigonal1']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

    def get_elastic_constants(self):
        A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(A2) / GPa
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

class Trigonal2(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C33', 'C44', 'C12', 'C13', 'C14', 'C15']
        strain_set = STRAIN_SETS['trigonal2']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
            [[0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0,-1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0,-1],
             [1,-1, 0, 0, 0, 0],
             [0, 0, 0,-1, 0, 0]],
            ])
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

    def get_elastic_constants(self):
        A2 = [self.fit_energy(s, e) for s, e in zip(self.strains, self.energies)]
        A2 = np.array(A2) / GPa
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

class Orthorhombic(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                      'C12', 'C13', 'C23']
        strain_set = STRAIN_SETS['orthorhombic']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

class Monoclinic(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                      'C12', 'C13', 'C23', 'C15', 'C25', 'C35', 'C46']
        strain_set = STRAIN_SETS['monoclinic']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

class Triclinic(Base):
    
    def get_ec_matrix(self):
        self.order = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66',
                      'C12', 'C13', 'C14', 'C15', 'C16', 'C23', 'C24', 'C25',
                      'C26', 'C34', 'C35', 'C36', 'C45', 'C46', 'C56']
        strain_set = STRAIN_SETS['triclinic']
        strain_matrix = strain_set[..., np.newaxis] * strain_set[:, np.newaxis]
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
        temp = strain_matrix[:, np.newaxis] * self.sym
        ec_matrix = np.sum(temp, axis=(2, 3))

        return ec_matrix

if __name__ == '__main__':
    #t = Cubic(a)
    energies = np.load('energies.npy')
    strains = np.load('strains.npy')
    #b = t.filter(a)
    t = Base.start(strains, energies, 'triclinic')
