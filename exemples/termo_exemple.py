import os
import numpy as np
import matplotlib.pyplot as plt

# import your datamanager class and the model you want to use
from exemples.TermoDataManager import TermoDataManager
from models.FuzzyModel import FuzzyTSModel

# Dataset import
path_to_data = os.path.join(os.path.dirname(__file__), '..', 'data')
data_files_name = ['Dados1_14a26_maio.txt', 'Dados2_14a26_maio.txt', 'Dados3_14a26_maio.txt', 'Dados4_Power6a10_14a26_maio.txt']

data_manager = TermoDataManager(path_to_data, data_files_name)
data_in, data_out = data_manager.get_data_in_out(verbose=True)

# there is a lot of data, for test we'll use only the first 10%
data_in = data_in[:]

# variable ranges
input_ranges = data_manager.get_dataframe_range(data_in)
output_range = data_manager.get_dataframe_range(data_out)[0]

# add a flexibility to ranges
margin = 10
input_ranges = [(interval[0]-margin, interval[1]+margin) for interval in input_ranges]
output_range = (output_range[0]-margin, output_range[1]+margin)

# variable names
input_names = data_in.columns
output_name = data_out.columns[0]

# N for each variable
n_list = [100]*2 + [60]*7

# splitting the data
Xt, yt, Xv, yv = data_manager.split_data(data_in, data_out, train_ratio=0.8)

# create the fuzzy model (VERIFICAR O N UTILIZADO NA DISSERTAÇÃO)
input_configs = [{"name":input_names[i],
                   "N":n_list[i], 
                   "range":input_ranges[i]} for i in range(len(input_names))]

output_config = {"name":output_name, "N":100, "range":output_range}

model = FuzzyTSModel(input_configs=input_configs, output_config=output_config) 

# visualizing the created variables 
# here we visualize only the first input variable 
# we are checking the pertinence values for the input 0.5
#var1 = model.var_manager.get(input_names[0])
#for term, value in var1.get_values(0.5).items():
#    print(term, ':',value)
#
## plot the variable
#var1.plot()
#
## model train
model.fit(Xt, yt)
#
### model prediction
y_pred = model.predict(Xv)
#
plt.plot(yv, label="Real")
plt.plot(y_pred, label="Fuzzy Pred")
plt.legend()
plt.show()
#
## Learned rules
print(model.explain())
