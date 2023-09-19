#=======================================================================================================================
#
#   TALLSorts v0 - The TALLSorts pipeline
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
#   Note: Inherited from Sklearn Pipeline
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from TALLSorts.common import message, root_dir

''' External '''
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class TALLSorts(Pipeline):

	"""
	Fundamentally, TALLSorts is just a pipeline, consisting of stages that need to be executed sequentially.

	This prepares the input for training or prediction. This TALLSorts class extends the Scikit Learn pipeline
	class and contains all the sequential stages needed to run TALLSorts.

	...

	Attributes
	__________
	Pipeline : Scikit-Learn pipeline class
		Inherit from this class.

	Methods
	-------
	transform(X)
		Execute every stage of the pipeline, transforming the initial input with each step. This does not include the
		final classification.
	clone
		Create an empty clone of this pipeline
	save(path="models/tallsorts.pkl.gz")
		Save pipeline in pickle format to the supplied path.

	"""

	def __init__(self, steps, memory=None, verbose=False, is_default_model=True):

		"""
		Initialise the class

		Attributes
		__________
		steps : list
			A list of all steps to be used in the pipeline (generally objects)
		memory : str or object
			Used to cache the fitted transformers. Default "None".
		verbose : bool
			Include extra messaging during the training of TALLSorts.
		"""

		self.steps = steps
		self.memory = memory
		self.verbose = verbose
		self.is_default_model = is_default_model
		self._validate_steps()

	def transform(self, X):

		"""
		Transform the input through the sequence of stages provided in self.steps.

		The final step (the classification stage) will NOT be executed here.
		This merely gets the raw input into the correct format.
		...

		Parameters
		__________
		X : Pandas DataFrame
			Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).

		Returns
		__________
		Xt : Pandas DataFrame
			Transformed counts.
		"""

		Xt = X.copy()
		for _, name, transform in self._iter(with_final=False):
			Xt = transform.transform(Xt)

		return Xt

	def clone(self):

		""" Create an empty copy of this pipeline.

		Returns
		_________
		A clone of this pipeline
		"""

		return clone(self)


	def save(self, path="models/tallsorts.pkl.gz"):

		""" Create an empty copy of this pipeline.

		Parameters
		__________
		path : str
			System path to save the picked object.

		Output
		_________
		Pickle the TALLSorts pipeline at the supplied path
		"""

		with open(path, 'wb') as output:
			joblib.dump(self, output, compress="gzip", protocol=-1)



