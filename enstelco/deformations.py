#!/usr/bin/env python
from enstelco.strains import STRAIN_SETS
from enstelco.utils import voigt_to_full, get_lattice_type
import elastic_coudert
from ase.io import read
import numpy as np
import os
import glob

crystal_familes = ['cubic', 'hexagonal', 'trigonal1',
                   'trigonal2', 'tetragonal1', 'tetragonal2',
                   'orthorhombic', 'monoclinic', 'triclinic']


class Idk:
    def __init__(self, atoms, calc=None, lattice_type=None, verbose=False):
        """

        Parameters
        ==========
        atoms:          relaxed ASE Atoms object to deform
        calc:           ASE calculator object to use for energy calculations
        lattice_type:   str/int, manually specify lattice_type either as
                        str (see crystal_families) or int (spacegroup number)
        verbose:        bool, print various summaries and display progress
        """

        self.atoms = atoms
        self.calc = calc
        self.verbose = verbose

        if lattice_type is None:
            self.lattice_type = get_lattice_type(atoms)
        elif isinstance(lattice_type, int):
            self.lattice_type = get_lattice_type(atoms, spacegroup=lattice_type)
        else:
            if lattice_type.lower() not in crystal_families:
                raise ValueError(f"Lattice type not recognized, choose from:\n\
                                   {crystal_families}")
            self.lattice_type = lattice_type.lower()

        self.strain_set = STRAIN_SETS[self.lattice_type]

    def deform(self, n=5, smin=0.0, smax=0.04, i_rerun=None, strains=None):
        if strains is None:
            self.strains = np.linspace(smin, smax, n)
        else:
            self.strains = np.array(strains)

        for i, eta  in enumerate(self.strain_set):
            if i_rerun is not None:
                if i != i_rerun:
                    continue
            eta = voigt_to_full(eta)

            for j, s in enumerate(self.strains):
                deformed_atoms = self.get_deformation(eta, s)
                path = f"{i:03d}/{j:03d}"
                os.makedirs(path, exist_ok=True)
                deformed_atoms.write(f"{path}/POSCAR")

            np.savetxt(f"{i:03d}/strains", self.strains)
                 

    def get_deformation(self, eta, strain):
        if len(eta) == 6:
            eta = voigt_to_full(eta)

        deformed_atoms = self.atoms.copy()
        A = np.identity(3) + eta * strain
        new_cell = np.dot(deformed_atoms.get_cell(), A)
        deformed_atoms.set_cell(new_cell, scale_atoms=True)
        return deformed_atoms

    def rerun(self, i_rerun, n, smin, smax, strains=None):
        os.system(f"rm -r {i_rerun:03d}/0*")
        self.deform(n=n, smin=smin, smax=smax, i_rerun=i_rerun, strains=strains)

        deformation_dirs = glob.glob(f'{i_rerun:03d}/0*')
        deformation_dirs.sort()
        
        for d in deformation_dirs:
            os.chdir(d)
            atoms = read("POSCAR")
            atoms.calc = self.calc
            atoms.get_potential_energy()
            atoms.write("opt.traj")
            os.chdir("../..")

        self.read_data()
        from enstelco.solve import Base
        self.base = Base.start(self.strains, self.energies, self.lattice_type)
        self.base.get_properties()


    def run(self, n=5, smin=0.0, smax=0.04, calc=None):
        self.deform(n=n, smin=smin, smax=smax)
        if calc is None:
            if self.calc is None:
                print("NO CALCULATOR PRESENT")
                print("ONLY PERFORMING DEFORMATIONS")
                return
            calc = self.calc
        deformation_dirs = glob.glob('0*/0*')
        deformation_dirs.sort()
        
        for d in deformation_dirs:
            os.chdir(d)
            atoms = read("POSCAR")
            atoms.calc = calc
            atoms.get_potential_energy()
            atoms.write("opt.traj")
            os.chdir('../..')

        self.read_data()
        from enstelco.solve import Base
        self.base = Base.start(self.strains, self.energies, self.lattice_type)
        self.base.get_properties()


    def read_data(self):
        n_sets = len(self.strain_set)
        strains = [np.loadtxt(f'{i:03d}/strains') for i in range(n_sets)]
        n_strains = len(strains[0])
        energies = [[read(f'{i:03d}/{j:03d}/opt.traj').get_potential_energy() for j in range(n_strains)] for i in range(n_sets)]
        ref_e = self.atoms.get_potential_energy()
        ref_V = self.atoms.get_volume()
        self.energies = (np.array(energies) - ref_e) / ref_V
        self.strains = np.array(strains)


    def thing(self):
        if not hasattr(self, 'energies'):
            self.read_data()

        from enstelco.solve import Base
        self.base = Base.start(self.strains, self.energies, self.lattice_type)
        self.base.get_properties()



if __name__ == '__main__':
    from ase.io import read
    from zeoml.lmp.calculator import DeepMD
    atoms = read('opt.traj')
    graph = '/home/sours/data/univ_ml/datasets/00_zeolites/tuning/all/final/64/graph.pb'
    calc = DeepMD(graph, minimize=True)
    idk = Idk(atoms, calc=calc)
    idk.thing()
