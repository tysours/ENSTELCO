#!/usr/bin/env python
import spglib
from numpy import array


def get_lattice_type(atoms, spacegroup=None, symprec=0.01):
    families = {
        2: "triclinic",
        15: "monoclinic",
        74: "orthorhombic",
        88: "tetragonal2",
        142: "tetragonal1",
        148: "trigonal2",
        167: "trigonal1",
        194: "hexagonal",
        230: "cubic",
    }

    if spacegroup is None:
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        s = spglib.get_spacegroup(cell, symprec=symprec)  # returns 'text (spacegroup)'
        spacegroup = int(s[s.find("(") + 1 : s.find(")")])

    for k, v in families.items():
        if spacegroup - k <= 0:
            return v


def voigt_to_full(V):
    xx, yy, zz, yz, xz, xy = V
    A = array(
        [[xx, 0.5 * xy, 0.5 * xz],
         [0.5 * xy, yy, 0.5 * yz],
         [0.5 * xz, 0.5 * yz, zz]]
    )
    return A


def full_to_voigt(A):
    A = 0.5 * (A + A.T)
    V = [A[0, 0], A[1, 1], A[2, 2], 2*A[1, 2], 2*A[0, 2], 2*A[0, 1]]
    return array(V)