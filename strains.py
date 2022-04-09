#!/usr/bin/env python
from numpy import array

def voigt_to_3x3(): 
    pass

all_strain_tensors = [
                      [1, 0, 0, 0, 0, 0], # C_11 0 
                      [0, 1, 0, 0, 0, 0], # C_22 1
                      [0, 0, 1, 0, 0, 0], # C_33 2
                      [0, 0, 0, 2, 0, 0], # C_44 3
                      [0, 0, 0, 0, 2, 0], # C_55 4
                      [0, 0, 0, 0, 0, 2], # C_66 5
                      [1, 1, 0, 0, 0, 0], # C_12 6
                      [1, 0, 1, 0, 0, 0], # C_13 7
                      [1, 0, 0, 2, 0, 0], # C_14 8
                      [1, 0, 0, 0, 2, 0], # C_15 9
                      [1, 0, 0, 0, 0, 2], # C_16 10
                      [0, 1, 1, 0, 0, 0], # C_23 11
                      [0, 1, 0, 2, 0, 0], # C_24 12
                      [0, 1, 0, 0, 2, 0], # C_25 13
                      [0, 1, 0, 0, 0, 2], # C_26 14
                      [0, 0, 1, 2, 0, 0], # C_34 15
                      [0, 0, 1, 0, 2, 0], # C_35 16
                      [0, 0, 1, 0, 0, 2], # C_36 17
                      [0, 0, 0, 2, 2, 0], # C_45 18
                      [0, 0, 0, 2, 0, 2], # C_46 19
                      [0, 0, 0, 0, 2, 2]  # C_56 20
              ] 

# NOTE: Monoclinic might need testing
#       Triclinic could be reduced to 18
STRAINS = {'cubic': all_strain_tensors[[0, 3, 6]],
           'hexa': all_strain_tensors[[0, 2, 3, 6, 7]],
           'tetra1': all_strain_tensors[[0, 2, 3, 5, 6, 7]],
           'tetra2': all_strain_tensors[[0, 2, 3, 5, 6, 7, 10]],
           'trigo1': all_strain_tensors[[0, 2, 3, 6, 7, 8]],
           'trigo2': all_strain_tensors[[0, 2, 3, 6, 7, 8, 9]],
           'ortho': all_strain_tensors[[0, 1, 2, 3, 4, 5, 6, 7, 11]],
           'mono': all_strain_tensors[[0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 19]],
           'tric': all_strain_tensors}
