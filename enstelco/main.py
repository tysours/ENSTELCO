import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
from enstelco.deformations import Deformations
from enstelco.solve import Solvers
from enstelco.utils import get_lattice_type

# XXX: Adds dynamic inheritance based on lattice type
#      (I should really rework solve.py to avoid this)
def dynamic_solver(cls):
    def make_instance(atoms, **kwargs):
        if kwargs.get('lattice_type'):
            lattice_type = kwargs.get('lattice_type')
        else:
            lattice_type = get_lattice_type(atoms)

        Solver = Solvers[lattice_type]

        class NewClass(cls, Solver):
            def __init__(self):
                cls.__init__(self, atoms, **kwargs)
                Solver.__init__(self)

        return NewClass()

    return make_instance


@dynamic_solver
class ENSTELCO(Deformations):
    def __init__(self, atoms, calc=None, lattice_type=None, verbose=False):
        Deformations.__init__(
            self, atoms, calc=calc, lattice_type=lattice_type, verbose=verbose
        )

    def read(self):
        n_sets = len(self.strain_set)
        strains = [np.loadtxt(f'{i:03d}/strains') for i in range(n_sets)]
        n_strains = len(strains[0])
        energies = [[read(f'{i:03d}/{j:03d}/opt.traj').get_potential_energy() for j in range(n_strains)] for i in range(n_sets)]
        ref_e = self.atoms.get_potential_energy()
        ref_V = self.atoms.get_volume()
        self.energies = (np.array(energies) - ref_e) / ref_V
        self.strains = np.array(strains)

    def process(self):
        if not self.energies:
            self.read()
        self.get_properties()

if __name__ == '__main__':
    from ase.io import read
    atoms = read('opt.traj')
    enstelco = ENSTELCO(atoms)
