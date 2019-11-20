import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display

path = "drug_consumption.xls"
seperator = ","
data_raw = pd.read_excel(path, sep=seperator)
display(data_raw)
