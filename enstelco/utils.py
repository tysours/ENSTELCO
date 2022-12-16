#!/usr/bin/env python
import spglib
from numpy import array


def get_lattice_type(atoms, spacegroup=None, symprec=0.01):
    """
    finds spacegroup and corresponding crystal family (e.g., cubic)
    of ASE Atoms object

    Parameters
    ----------
    atoms
        ASE Atoms object to find lattice type

    spacegroup : int
        option to manually specify spacegroup

    symprec : float
        spglib symmetry tolerance to find spacegroup higher value for less
        symmetric structures

    Returns
    -------
    str
        lattice type of atoms
    """
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
    """
    find full 3x3 array from 6 element vector in voigt notation

                                    [[xx, xy, xz],
    [xx, yy, zz, yz, xz, xy]  -->    [xy, yy, yz],
                                     [xz, yz, zz]]



    Parameters
    ----------
    V : array-like
        6 element vector in voigt notation

    Returns
    -------
    array
        3x3 array of stress tensor components
    """


    xx, yy, zz, yz, xz, xy = V
    A = array(
        [[xx, 0.5 * xy, 0.5 * xz],
         [0.5 * xy, yy, 0.5 * yz],
         [0.5 * xz, 0.5 * yz, zz]]
    )
    return A


def full_to_voigt(A):
    """
    find full (symmetric) 3x3 array from 6 element vector in voigt notation

    [[xx, xy, xz],
     [xy, yy, yz],  -->   [xx, yy, zz, yz, xz, xy]
     [xz, yz, zz]]

    Parameters
    ----------
    A : array-like
        3x3 array to transform into voigt notation vector

    Returns
    -------
    array
        6 element vector in voigt notation
    """
    A = 0.5 * (A + A.T)
    V = [A[0, 0], A[1, 1], A[2, 2], 2*A[1, 2], 2*A[0, 2], 2*A[0, 1]]
    return array(V)


#seaborn.color_palette('deep')
colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
          (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
          (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
          (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
          (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
          (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
          (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
          (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
          (0.8, 0.7254901960784313, 0.4549019607843137),
          (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]


def _generate_color():
    for color in colors:
        yield color


_gen_color = _generate_color()
def next_color():
    """
    Convenience function to grab a new color during plotting loops.

    Returns
    -------
    tuple[float]
        New color rgb vals from seaborn.color_palette('deep')

    Example
    -------
        .. code-block:: python

            for values in all_values:
                plt.plot(values, color=next_color())
    """
    return next(_gen_color)
