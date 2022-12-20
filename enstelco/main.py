import os
import numpy as np
import matplotlib.pyplot as plt
from ase import io
from ase.units import GPa
from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from enstelco.deformations import Deformations
from enstelco.solve import Solvers
from enstelco.utils import get_lattice_type, colors
from enstelco.strains import STRAIN_SETS


# XXX: Adds dynamic inheritance based on lattice type
#      (I should really rework solve.py to avoid this)
def dynamic_solver(cls):
    def make_instance(atoms, **kwargs):
        if kwargs.get('lattice_type'):
            lattice_type = kwargs.get('lattice_type')
        else:
            lattice_type = get_lattice_type(atoms)

        Solver = Solvers[lattice_type]

        class RealENSTELCO(cls, Solver):
            def __init__(self):
                cls.__init__(self, atoms, **kwargs)
                Solver.__init__(self)

        return RealENSTELCO()

    return make_instance


@dynamic_solver
class ENSTELCO(Deformations):
    def __init__(self, atoms, calc=None, lattice_type=None, verbose=False,
                 input_file='POSCAR', output_file='opt.traj'):

        Deformations.__init__(
            self, atoms, calc=calc, lattice_type=lattice_type, verbose=verbose,
            input_file=input_file, output_file=output_file,
        )

    def read(self, opt_file='opt.traj'):
        n_sets = len(self.strain_set)
        strains = [np.loadtxt(f'{i:03d}/strains') for i in range(n_sets)]
        n_strains = len(strains[0])

        def get_E(path):
            return io.read(os.path.join(path, opt_file)).get_potential_energy()

        energies = [[get_E(f"{i:03d}/{j:03d}") for j in range(n_strains)]
                    for i in range(n_sets)]

        self._ref_e = self.atoms.get_potential_energy()
        self._ref_V = self.atoms.get_volume()
        self.energies = (np.array(energies) - self._ref_e) / self._ref_V
        self.strains = np.array(strains)

    def process(self):
        if self.energies is None:
            self.read()
        self.get_properties()

    def summarize(self, boring=False, file_name=None):
        if not hasattr(self, 'properties'):
            self.process()

        if file_name is not None:
            boring = True

        ec_panel = self._write_ec_panel(boring=boring)
        tensor_panel = '' if boring else self._write_tensor_panel()
        properties_panel = self._write_properties_panel(boring=boring)

        if file_name is not None:
            with open(file_name, 'w') as file:
                file.write(f'{ec_panel}{properties_panel}')
            return

        console = Console()
        display = Group(ec_panel, tensor_panel, properties_panel)
        print()
        console.print(display)
        print()

    def _write_ec_panel(self, boring=False):
        if boring:
            text = ''
            for k, v in self.elastic_constants.items():
                text += f'{k} {v}\n'
            return text

        ec_tables = []
        ec_vals = []
        for i, (k, v) in enumerate(self.elastic_constants.items()):
            if i % 7 == 0:
                ec_tables.append(Table())
                ec_vals.append([])
            ec_tables[-1].add_column(k, justify='center')
            ec_vals[-1].append(round(v, 1))

        for vals, table in zip(ec_vals, ec_tables):
            table.add_row(*list(map(str, vals)))

        ec_panel = Panel(Group(*ec_tables), title='ELASTIC CONSTANTS', expand=False)
        return ec_panel

    def _write_tensor_panel(self):
        tensor_table = Table.grid()
        for i in range(6):
            tensor_table.add_column(justify='center', min_width=8)
        for j in range(6):
            vals = np.round(self.elastic_tensor[j, :], 1)
            tensor_table.add_row(*list(map(str, vals)))

        tensor_panel = Panel(tensor_table, title='ELASTIC TENSOR', expand=False)
        return tensor_panel

    def _write_properties_panel(self, boring=False):
        if boring:
            text = ''
            def append_line(text, attr):
                text += f'{attr} {getattr(self, attr)}\n'
                return text

            attributes = [
                'K_V', 'K_R', 'K_VRH',
                'E_V', 'E_R', 'E_VRH',
                'G_V', 'G_R', 'G_VRH',
                'v_V', 'v_R', 'v_VRH',
            ]

            for a in attributes:
                text = append_line(text, a)
            return text

        headers = ['Method', 'Bulk modulus', "Young's modulus", 'Shear modulus',
                "Poisson's ratio"]
        properties_table = Table(*headers)

        mech_keys = ['Voigt', 'Reuss', 'Hill']
        for key, vals in zip(mech_keys, self.properties):
            properties_table.add_row(key, *list(map(str, np.round(vals, 2))))

        title = 'MECHANICAL PROPERTIES'
        subtitle = 'Calculated with ELATE: https://progs.coudert.name/elate'
        properties_panel = Panel(properties_table, title=title,
                                 subtitle=subtitle, expand=False)
        return properties_panel

    def plot(self, n_max=4, axs=None, save_file=None):
        etas = [i for i, _ in enumerate(STRAIN_SETS[self.lattice_type])]

        n_panels = int(np.ceil(len(etas) / n_max))
        n_cols = n_panels if n_panels < 4 else 4
        n_rows = int(np.ceil(n_panels / 4))
        width = 0.5 + 3.5 * n_cols
        height = 3 * n_rows

        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
        if n_cols == n_rows == 1:
            axs = np.array([axs])
        if axs.ndim == 1:
            axs = [axs]
        k = 0
        leg_params = dict(labelcolor='linecolor', frameon=0,
                handlelength=0.5, fontsize=12)
        for a in range(n_rows):
            for b in range(n_cols):
                for n in range(n_max):
                    self.plot_eta_i(etas[k], color=colors[n], ax=axs[a][b])
                    k += 1
                    if k == len(etas):
                        break
                axs[a][b].legend(**leg_params)
                axs[a][b].tick_params(labelsize=12)
                if b == 0:
                    axs[a][b].set_ylabel('Energy', fontsize=14)
                if a + 1 == n_rows:
                    axs[a][b].set_xlabel('Strain (%)', fontsize=14)

        plt.tight_layout()
        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()

    def plot_eta_i(self, i, color=None, ax=None):
        if ax is None:
            ax = plt.gca()

        data, fit_data = self._get_plot_data(i)
        text = f'$\eta$$_{i}$'
        ax.plot(*fit_data, ls='-', lw=2.0, color=color, label=text)
        ax.plot(*data, ls='', marker='o', mew=1, mec='black', color=color, ms=7)

    def _get_plot_data(self, i_read):
        A2 = self._A2[i_read]

        strains = self.strains[i_read]
        energies = self.energies[i_read] * self._ref_V + self._ref_e

        fit_strains = np.linspace(min(strains), max(strains), 30)
        fit_energies = self.get_E(A2, strains=fit_strains) * self._ref_V + self._ref_e

        return (strains * 100, energies), (fit_strains * 100, fit_energies)


if __name__ == '__main__':
    from ase.io import read
    atoms = read('opt.traj')
    enstelco = ENSTELCO(atoms)
    enstelco.read()
    enstelco.plot_cij(11)
