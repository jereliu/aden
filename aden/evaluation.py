__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pickle as pk
import pandas as pd

import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

trace = pk.load(open("./data/ensemble_logistic.pkl", "rb"))

# trace = pk.load(open("./data/ensemble_dirichlet.pkl", "rb"))


model_name = ["Linear", "Poly2", "Poly3", "Poly4",
              "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
              "MLP_ARD", "SpecMix"]
