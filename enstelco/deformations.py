#!/usr/bin/env python
import numpy as np
import os
import glob
from ase.io import read

from enstelco.strains import STRAIN_SETS
from enstelco.utils import voigt_to_full, get_lattice_type

crystal_families = ['cubic', 'hexagonal', 'trigonal1',
                    'trigonal2', 'tetragonal1', 'tetragonal2',
                    'orthorhombic', 'monoclinic', 'triclinic']


class Deformations:
    def __init__(self, atoms, calc=None, lattice_type=None, verbose=False,
            input_file='POSCAR', output_file='opt.traj'):
        """

        Parameters
        ----------
        atoms
            relaxed ASE Atoms object to deform.

        calc
            ASE calculator object to use for energy calculations.

        lattice_type : (str or int)
            manually specify lattice_type either as str (see crystal_families)
            or int (spacegroup number).

        verbose : bool
            print various summaries and display progress if True.

        input_file : str
            name of input file to write for each deformation.

        output_file : str
            name of output file to write for each deformation calculation.
        """

        self.atoms = atoms
        self.calc = calc
        self.verbose = verbose
        self.input_file = input_file
        self.output_file = output_file

        if lattice_type is None:
            self.lattice_type = get_lattice_type(atoms)
        elif isinstance(lattice_type, int):
            self.lattice_type = get_lattice_type(atoms, spacegroup=lattice_type)
        else:
            if lattice_type.lower() not in crystal_families:
                raise ValueError(f'Lattice type not recognized, choose from:\n\
                                   {crystal_families}')
            self.lattice_type = lattice_type.lower()

        self.strain_set = STRAIN_SETS[self.lattice_type]

    def deform(self, n=5, smin=0.0, smax=0.04, i_rerun=None, strains=None):

        if strains is None:
            self.strains = np.linspace(smin, smax, n)
        else:
            self.strains = np.array(strains)

        self._dirs = []
        for i, eta  in enumerate(self.strain_set):
            if i_rerun is not None:
                if i != i_rerun:
                    continue
            eta = voigt_to_full(eta)

            for j, s in enumerate(self.strains):
                deformed_atoms = self.get_deformation(eta, s)
                path = os.path.join(f'{i:03d}', f'{j:03d}')
                os.makedirs(path, exist_ok=True)
                deformed_atoms.write(os.path.join(path, self.input_file))
                self._dirs.append(path)

            np.savetxt(os.path.join(f'{i:03d}', 'strains'), self.strains)

    def get_deformation(self, eta, strain):
        if len(eta) == 6:
            eta = voigt_to_full(eta)

        deformed_atoms = self.atoms.copy()
        A = np.identity(3) + eta * strain
        new_cell = np.dot(deformed_atoms.get_cell(), A)
        deformed_atoms.set_cell(new_cell, scale_atoms=True)
        return deformed_atoms

    def rerun(self, i_rerun, n, smin, smax, strains=None, deform=True):
        """
        Rerun specific deformation. Might be useful if certain directions give
        poor fits and you want to manually adjust strains for that direction.
        Maybe poor practice? :shrug:
        """
        os.system(f'rm -r {i_rerun:03d}/0*')
        if deform:
            self.deform(n=n, smin=smin, smax=smax, i_rerun=i_rerun, strains=strains)

        deformation_dirs = glob.glob(f'{i_rerun:03d}/0*')
        deformation_dirs.sort()

        for d in deformation_dirs:
            os.chdir(d)
            atoms = read(self.input_file)
            atoms.calc = self.calc
            atoms.get_potential_energy()
            atoms.write(self.output_file)
            os.chdir('../..')

    def run(self, n=5, smin=0.0, smax=0.04, calc=None, deform=True):
        if deform:
            self.deform(n=n, smin=smin, smax=smax)
        if calc is None:
            if self.calc is None:
                print('NO CALCULATOR PRESENT')
                print('ONLY PERFORMING DEFORMATIONS')
                return
            calc = self.calc
        if not hasattr('self', '_dirs'):
            # user might make their own dirs for whatever reason
            deformation_dirs = sorted(glob.glob('0*/0*'))
        else:
            deformation_dirs = self._dirs

        for d in deformation_dirs:
            os.chdir(d)
            atoms = read(self.input_file)
            atoms.calc = calc
            atoms.get_potential_energy()
            atoms.write(self.output_file)
            os.chdir('../..')


if __name__ == '__main__':
    from ase.io import read
    from zeoml.lmp.calculator import DeepMD
    atoms = read('opt.traj')
    graph = '/home/sours/data/univ_ml/datasets/00_zeolites/tuning/all/final/64/graph.pb'
    calc = DeepMD(graph, minimize=True)
    idk = Deformations(atoms, calc=calc)
    idk.process()
