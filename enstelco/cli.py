import argparse

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
    """
    help_info = 'Deform input structure and write corresponding deformed inputs'
    def add_args(self):
        pass

    def main(self, args):
        raise NotImplementedError('Coming soon...')


class ProcessCLI(Base):
    """
    """
    help_info = ''
    def add_args(self):
        pass

    def main(self, args):
        raise NotImplementedError('Coming soon...')


class PlotCLI(Base):
    """
    """
    help_info = ''
    def add_args(self):
        pass

    def main(self, args):
        raise NotImplementedError('Coming soon...')


cli_parsers = {
        'deform': DeformCLI,
        'process': ProcessCLI,
        'plot': PlotCLI,
    }


def main():
    parser = argparse.ArgumentParser(prog='enstelco',
                                     description='ENergy-STrain ELastic COnstants'\
                                             ' and mechanical property calculations!',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     )
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(title='commands', dest='command')

    command_clis = {}
    for comm in commands:
        CLI = cli_parsers[comm]
        doc = CLI.__doc__
        subparser = subparsers.add_parser(comm, help=CLI.help_info, description=doc,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cli = CLI(subparser)
        cli.add_args()
        command_clis[comm] = cli

    parsed_args = parser.parse_args()
    command_clis[parsed_args.command].main(parsed_args)
