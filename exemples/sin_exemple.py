import os
import numpy as np
import matplotlib.pyplot as plt
from models.FuzzyModel import FuzzyTSModel

# ====================== Dataset creation ======================: 
# Using a sine wave with 3 shifts to the left as input variables
# and the original sine wave as output variable
# The model will learn to predict the sine wave based on its past values
# (time series forecasting)
t = np.arange(0,8*np.pi,0.02)

# the signal is a sine wave with a change in amplitude at the middle of the signal
y1 = np.sin(t[:len(t)//2])
y2 = -np.sin(t[len(t)//2:len(t)//2+len(t)//3])
exp_arg = -(t[len(t)//2+len(t)//3:] - t[len(t)//2+len(t)//3])
y3 = -np.exp(exp_arg)*np.sin(t[len(t)//2+len(t)//3:])  
y = np.concatenate((y1, y2,y3))   

# 3 shifts to the left
x1 = np.roll(y, 1)
x2 = np.roll(y, 2)
x3 = np.roll(y, 3)
x1, x2, x3,  y = x1[3:], x2[3:], x3[3:], y[3:]

# division in train and validation data:
train_size = int(len(t)*0.5)
x1t, x2t, x3t, yt = x1[:train_size], x2[:train_size], x3[:train_size], y[:train_size]
Xt = np.vstack((x1t, x2t, x3t))
Xt = Xt.T
x1v, x2v, x3v, yv = x1[train_size:], x2[train_size:], x3[train_size:] ,y[train_size:]
Xv = np.vstack((x1v, x2v, x3v))
Xv = Xv.T
plt.plot(t[3:train_size+3], yt, label="Train")
plt.plot(t[train_size+3:], yv, label="Validation")
plt.legend()
plt.title("Train and Validation data")
plt.grid()
plt.show()


# ====================== Fuzzy model creation and usage ======================:

# variable ranges
input_range = [-1.5, 1.5]

input_configs = [{"name":f"var{i+1}","N":6, "range":input_range} for i in range(3)]
output_config = {"name":"out","N":6, "range":input_range}

# create the fuzzy model
model = FuzzyTSModel(input_configs=input_configs, output_config=output_config)

# visualizing the created variables 
# here we visualize only the first input variable 
# we are checking the pertinence values for the input 0.5
var1 = model.var_manager.get('var1')
for term, value in var1.get_values(0.5).items():
    print(term, ':',value)

# plot the variable
var1.plot()

# model train
model.fit(Xt, yt)

#################################################################################################
## Here we're saving and loading the model just for exemple usage, but it's not necessary to do it
# you can load a model in a new code and not  
# saving the model
model_name = 'fuzzyTS_sin_exemple'
path_to_save = os.path.join(os.getcwd(), 'exemples', 'saved_fuzzy_models')
os.makedirs(path_to_save, exist_ok=True)
path_to_save = os.path.join(path_to_save, model_name)
model.save(path_to_save)

# loading model
path_to_load = path_to_save
model.load(path_to_load)
#################################################################################################

# pruning parameters
model.rule_manager.prune_weight_threshold = 0.1
model.rule_manager.prune_use_threshold = 2
model.rule_manager.prune_window = 25

# model prediction
y_pred = model.predict(Xv)
error = model.metrics.absolute_percentage_error(yv, y_pred)

fig, ax = plt.subplots(2,1)
fig.suptitle("Validation data: Real vs Fuzzy Predicted")
ax[0].plot(yv, label="Real")
ax[0].plot(y_pred, label="Fuzzy Pred")
ax[0].grid(True)
ax[1].plot(error)
ax[1].set_ylabel("abs error %")
ax[1].grid(True)
plt.legend()
plt.show()

# Metrics
results = model.get_score(y_pred=y_pred, y_true=yv)
print(f"MAE: {results['MAE']}\nMAPE: {results['MAPE']}\nRMSE: {results['RMSE']}\nR2: {results['R2']}")


# Learned rules
#print(model.explain())
