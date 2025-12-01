import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import re
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np


data = pd.read_csv('~/Documents/Data_Wrangling/Fifa_world_cup_matches.csv')

data.columns

data.info

data.info(verbose=True)




