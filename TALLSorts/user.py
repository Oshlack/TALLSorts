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

            ''' Mode '''
            self.mode = self.input.mode if self.input.mode else 'test'

            ''' Data '''
            self.samples = self.input.samples
            # self.counts = False if not self.input.counts else True

            ''' Training mode arguments '''
            self.sample_sheet = self.input.sample_sheet if self.input.sample_sheet else None
            self.hierarchy = self.input.hierarchy if self.input.hierarchy else None
            self.training_params = self.input.training_params if self.input.training_params else None
            self.training_cores = self.input.training_cores if self.input.training_cores else 1
            self.model_path = self.input.model_path if self.input.model_path else None

            '''Gene labels'''
            self.gene_labels = self.input.gene_labels if self.input.gene_labels else 'id'

            self.destination = False if not self.input.destination else self.input.destination

            '''Misc'''
            self._input_checks()
            self._load_samples()
            self._load_sample_sheet()
            self._load_hierarchy()
            self._load_training_params()

        else:
            message("No arguments supplied. Please use tallsorts --help for further information about input.")
            sys.exit(0)

    def _is_cli(self):
        return len(sys.argv) > 1

    def _get_args(self):

        ''' Get arguments and options from CLI '''

        cli = argparse.ArgumentParser(description="TALLSorts CLI")

        cli.add_argument('--mode', '-m',
                         required=False,
                         help=("""Run TALLSorts in model-training mode."""))

        cli.add_argument('--samples', '-s',
                         required=True,
                         help=("""Path to samples (rows) x genes (columns) csv file representing a raw counts matrix."""))

        cli.add_argument('--gene-labels', '--gl',
                         required=False,
                         help=("""Whether your gene labels are in Ensembl ID or Gene Symbol form. Defaults to Ensembl ID.
                               'id' (default): Ensembl ID
                               'symbol': Gene symbol"""))

        cli.add_argument('--destination', '-d',
                         required=False,
                         help=("""Path to where you want the results (if in testing mode) or TALLSorts model (if in training mode) to be saved."""))
                
        cli.add_argument('--model-path', '--mp',
                         required=False,
                         help=("""Path to TALLSorts-generated model file."""))

        # testing arguments        
        cli.add_argument('--sample-sheet', '--ss',
                         required=False,
                         help=("""Path to a binary 0/1 sample sheet with samples (rows) and labels (columns)"""))
        
        cli.add_argument('--hierarchy',
                         required=False,
                         help=("""Path to a sheet containing hierarchy information for labels."""))
        
        cli.add_argument('--training-params', '--tp',
                         required=False,
                         help=("""Path to a sheet containing parameters to feed into sklearn's LogisticRegression model."""))
        
        cli.add_argument('--training-cores', '--tc',
                         required=False,
                         help=("""Number of parallel cores to use during training. Defaults to 1."""))


        # cli.add_argument('--counts',
        #                  required=False,
        #                  action="store_true",
        #                  help=("""(bool, default=False) Output preprocessed counts (normalised and standardised)."""))

        user_input = cli.parse_args()
        return user_input

    def _input_checks(self):
        
        
        if self.gene_labels and self.gene_labels not in ('id', 'symbol'):
            message("Error: Please supply a valid argument for --gene-labels (either 'id' or 'symbol'). Exiting.")
            sys.exit()

        if self.mode == 'train':
            if not self.sample_sheet:
                message("Error: Please supply a valid path for --sample-sheet (path to sample sheet). Exiting.")
                sys.exit()
            if not self.hierarchy:
                message("Error: Please supply a valid path for --hierarchy (path to hierarchy sheet). Exiting.")
                sys.exit()
            if not self.destination:
                message("Error: a directory (-d /path/to/output/) is required for storing the custom TALLSorts model. Exiting.")
                sys.exit()
        elif self.mode == 'test':
            if self.sample_sheet:
                message("Error: TALLSorts is in testing mode, but --sample-sheet was provided. Did you forget '--mode train'? Exiting.")
                sys.exit()
            if self.hierarchy:
                message("Error: TALLSorts is in testing mode, but --hierarchy was provided. Did you forget '--mode train'? Exiting.")
                sys.exit()
            if self.training_params:
                message("Error: TALLSorts is in testing mode, but --training-params, was provided. Did you forget '--mode train'? Exiting.")
                sys.exit()
            if self.training_cores is not None and self.training_cores != 1:
                message("Error: TALLSorts is in testing mode, but --training-cores was provided. Did you forget '--mode train'? Exiting.")
                sys.exit()
            
            if not self.destination:
                message("Error: a directory (-d /path/to/output/) is required for storing TALLSorts results. Exiting.")
                sys.exit()

        try:
            self.training_cores = int(self.training_cores)
        except:
            message("Error: Please provide an integer value for --training-cores. Exiting.")
            sys.exit()

    def _load_samples(self):
        if self.samples:
            self.samples = pd.read_csv(self.samples, index_col=0, header=0)

    def _load_sample_sheet(self):
        if self.sample_sheet:
            self.sample_sheet = pd.read_csv(self.sample_sheet, index_col=0, header=0)
            self.sample_sheet.fillna(0, inplace=True)

    def _load_hierarchy(self):
        if self.hierarchy:
            self.hierarchy = pd.read_csv(self.hierarchy, index_col=0, header=0)
            self.hierarchy.fillna('', inplace=True)

    def _load_training_params(self):
        if self.training_params:
            self.training_params = pd.read_csv(self.training_params, index_col=0, header=0)
            self.training_params.fillna('', inplace=True)