#=======================================================================================================================
#
#   TALLSorts v0 - Command Line Interface
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
#   Parse user arguments
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from TALLSorts.common import message, root_dir

''' External '''
import sys, argparse
import pandas as pd
import ast

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class UserInput:

    def __init__(self):
        if self._is_cli():
            self.cli = True
            self.input = self._get_args()

            ''' Data '''
            self.samples = self.input.samples
            self.counts = False if not self.input.counts else True

            '''Prediction Parameters '''
            self.destination = False if not self.input.destination else self.input.destination

            '''Misc'''
            self._input_checks()
            self._load_samples()

        else:
            message("No arguments supplied. Please use tallsorts --help for further information about input.")
            sys.exit(0)

    def _is_cli(self):
        return len(sys.argv) > 1

    def _get_args(self):

        ''' Get arguments and options from CLI '''

        cli = argparse.ArgumentParser(description="TALLSorts CLI")
        cli.add_argument('-samples', '-s',
                         required=True,
                         help=("""Path to samples (rows) x genes (columns) csv file representing a raw counts matrix.

                                  Note: hg19 only supported currently, use other references at own risk."""))

        cli.add_argument('-destination', '-d',
                         required=False,
                         help=("""Path to where you want the final report to be saved."""))

        cli.add_argument('-counts',
                         required=False,
                         action="store_true",
                         help=("""(bool, default=False) Output preprocessed counts (normalised and standardised)."""))

        user_input = cli.parse_args()
        return user_input

    def _input_checks(self):
        if not self.destination:
            message("Error: a destination (-d /path/to/output/) is required. Exiting.")
            sys.exit()

    def _load_samples(self):

        if self.samples:
            self.samples = pd.read_csv(self.samples, index_col=0, header=0)