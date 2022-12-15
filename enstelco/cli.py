import argparse
from textwrap import dedent

from enstelco import __version__, ENSTELCO

commands = (
    'deform',
    'process',
    'plot',
)

class Base:
    def __init__(self, parser):
        self.parser = parser


class DeformCLI(Base):
    """
    Perform deformations on input structure with various strains to
    calculate energies and fit to elastic constants. Number of deformations
    is dependent on lattice symmetry (automatically determined or can be
    specified).

    Examples:

        $ enstelco deform CHA.cif
        $ enstelco deform --lo -0.02 --hi 0.02 thing.xyz
        $ enstelco deform -l cubic cubic_thing_but_improperly_detected_automatically.xyz
    """
    help_info = 'Deform input structure and write corresponding deformed inputs'
    def add_args(self):
        self.parser.add_argument('structure', type=str,
            help='Undeformed structure to be used for deformations'\
                    ' (any ASE-readable format)')

        self.parser.add_argument('-n', type=int, default=5,
                help='Number of strains to apply between --lo and --hi')
        self.parser.add_argument('--lo', type=float, default=0.0,
                help='Lower strain bound')
        self.parser.add_argument('--hi', type=float, default=0.04,
                help='Upper strain bound')
        self.parser.add_argument('-i', '--input_file', type=str, default='POSCAR',
                help='Name (and ext) of input file to write for each deformation')
        self.parser.add_argument('-l', '--lattice-type', type=str, default=None,
                help='Specific lattice type to force (automatically determined by'\
                        ' default). Can be spacegroup number of choose from '\
                        'enstelco.deformations.crystal_families')

    def main(self, args):
        from ase.io import read
        atoms = read(args.structure)
        enstelco = ENSTELCO(atoms, lattice_type=args.lattice_type,
                            input_file=args.input_file)
        enstelco.deform(n=args.n, smin=args.lo, smax=args.hi)


class ProcessCLI(Base):
    """
    Calculate elastic constants and mechanical properties from completed
    deformation calculations.
    """
    help_info = 'Calculate elastic constants + mechanical properties'
    def add_args(self):
        pass

    def main(self, args):
        raise NotImplementedError('Coming soon...')


class PlotCLI(Base):
    """
    Plot energy-strain data from completed calculations with second order fits
    and annotated elastic constant values.
    """
    help_info = 'Plot energy-strain data and fits'
    def add_args(self):
        pass

    def main(self, args):
        raise NotImplementedError('Coming soon...')


cli_parsers = {
    'deform': DeformCLI,
    'process': ProcessCLI,
    'plot': PlotCLI,
}


class MyFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    HelpFormatter that retains docstring formatting (mostly to preserve example
    listing).
    """

    def _fill_text(self, text, width, indent):
        return dedent(text)


def main():
    parser = argparse.ArgumentParser(prog='enstelco',
                                     description='ENergy-STrain ELastic COnstants'\
                                             ' and mechanical property calculations!',
                                     formatter_class=MyFormatter,
                                     )
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(title='commands', dest='command')

    command_clis = {}
    for comm in commands:
        CLI = cli_parsers[comm]
        doc = CLI.__doc__
        subparser = subparsers.add_parser(comm, help=CLI.help_info, description=doc,
                formatter_class=MyFormatter)
        cli = CLI(subparser)
        cli.add_args()
        command_clis[comm] = cli

    parsed_args = parser.parse_args()
    command_clis[parsed_args.command].main(parsed_args)
