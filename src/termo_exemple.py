import os
import numpy as np
import matplotlib.pyplot as plt
from fuzzymodel import FuzzyTSModel
from TermoDataManager import TermoDataManager

# Dataset import
path_to_data = os.path.join(os.path.dirname(__file__), '..', 'data')
data_files_name = ['Dados1_14a26_maio.txt', 'Dados2_14a26_maio.txt', 'Dados3_14a26_maio.txt', 'Dados4_Power6a10_14a26_maio.txt']

data_manager = TermoDataManager(path_to_data, data_files_name)
data_in, data_out = data_manager.get_data_in_out(verbose=True)

# variable ranges
input_range = [-1.5, 1.5]
output_range = input_range

# variable names
input_names = data_in.columns
output_name = data_out.columns[0]

# splitting the data
Xt, yt, Xv, yv = data_manager.split_data(data_in, data_out, train_ratio=0.8)

# create the fuzzy model
model = FuzzyTSModel(input_names=input_names, output_name=output_name, num_regions=7,
                          input_range=input_range, output_range=output_range)

# visualizing the created variables 
# here we visualize only the first input variable 
# we are checking the pertinence values for the input 0.5
#var1 = model.var_manager.get('var1')
#for term, value in var1.get_values(0.5).items():
#    print(term, ':',value)
#
## plot the variable
#var1.plot()
#
## model train
#model.fit(Xt, yt)
#
### model prediction
#y_pred = model.predict(Xv)
#
#plt.plot(yv, label="Real")
#plt.plot(y_pred, label="Fuzzy Pred")
#plt.legend()
#plt.show()
#
## Learned rules
#print(model.explain())
