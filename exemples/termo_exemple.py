import os
import numpy as np
import matplotlib.pyplot as plt

# import your datamanager class and the model you want to use
from exemples.TermoDataManager import TermoDataManager
from models.FuzzyModel import FuzzyTSModel

MAX_RULES = 200
N_FUZZYSETS_LIST = [10,5]

def prepare_data():
    # Dataset import
    path_to_data = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_files_name = ['Dados1_14a26_maio.txt', 'Dados2_14a26_maio.txt', 'Dados3_14a26_maio.txt', 'Dados4_Power6a10_14a26_maio.txt']

    data_manager = TermoDataManager(path_to_data, data_files_name)
    data_in, data_out = data_manager.get_data_in_out(verbose=True)

    # there is a lot of data, for test we'll use only the first 10%
    use_data_percentage = 0.05
    nb_of_samples = int(data_in.shape[0] * use_data_percentage)
    print(f"Using only {nb_of_samples} of {data_in.shape[0]} samples")
    data_in = data_in[:nb_of_samples]
    data_out = data_out[:nb_of_samples]

    # variable ranges
    input_ranges = data_manager.get_dataframe_range(data_in)
    output_range = data_manager.get_dataframe_range(data_out)[0]

    # add a flexibility to ranges
    margin = 0.1 # 10%
    input_ranges = [(interval[0]-margin*interval[0], interval[1]+margin*interval[1]) for interval in input_ranges]
    output_range = (output_range[0]-margin*output_range[0], output_range[1]+margin*output_range[1])

    # variable names
    input_names = data_in.columns
    print(input_names)
    output_name = data_out.columns[0]

    # N for each variable
    n_list = [N_FUZZYSETS_LIST[0]]*2 + [N_FUZZYSETS_LIST[1]]*7

    # splitting the data
    Xt, yt, Xv, yv = data_manager.split_data(data_in, data_out, train_ratio=0.95)
    Xt, yt, Xv, yv = np.asarray(Xt), np.asarray(yt).ravel(), np.asarray(Xv), np.asarray(yv).ravel()

    # create the fuzzy model (VERIFICAR O N UTILIZADO NA DISSERTAÇÃO)
    input_configs = [{"name":input_names[i],
                       "N":n_list[i], 
                       "range":input_ranges[i]} for i in range(len(input_names))]

    output_config = {"name":output_name, "N":N_FUZZYSETS_LIST[0], "range":output_range}
    return input_configs, output_config, Xt, yt, Xv, yv

def train_save_model(input_configs, output_config, Xt, yt):
    model = FuzzyTSModel(input_configs=input_configs, output_config=output_config, max_rules=int(MAX_RULES)) 

    ## model train
    model.fit(Xt, yt)

    model_name = 'fuzzyTS_termo_exemple'
    path_to_save = os.path.join(os.getcwd(), 'exemples', 'saved_fuzzy_models')
    os.makedirs(path_to_save, exist_ok=True)
    path_to_save = os.path.join(path_to_save, model_name)
    model.save(path_to_save)

def load_predict_model(Xv, yv):
    # loading model
    path_to_load = os.path.join(os.getcwd(), 'exemples', 'saved_fuzzy_models', 'fuzzyTS_termo_exemple' ) 
    model = FuzzyTSModel.load(path_to_load)
    
    ### model prediction
    y_pred = model.predict(Xv)

    y_error = model.metrics.absolute_percentage_error(yv, y_pred)

    #
    fig, ax = plt.subplots(2,1)
    ax[0].plot(yv, color='red', label="Dado Bruto")
    ax[0].plot(y_pred, linestyle='--', color='blue', label="Simulação")
    ax[0].set_ylims([0,1])
    ax[0].legend()
    ax[1].plot(y_error, linestyle='--', color='black', label="Absolute Percentual Error")
    ax[1].set_ylims([0,100])
    ax[1].set_ylabel("Absolute Error (%)")

    # saving plot
    img_path = os.path.join(os.getcwd(), 'exemples', 'error_figs')
    os.makedirs(img_path, exist_ok=True)
    img_name = f"old_fuzzy_error_{N_FUZZYSETS_LIST[0]}_{N_FUZZYSETS_LIST[1]}_max_rules{MAX_RULES}"
    plt.savefig(os.path.join(img_path,img_name+'.png'))
    #
    ## Learned rules
    #print(model.explain())


if __name__ == '__main__':
    input_configs, output_config, Xt, yt, Xv, yv = prepare_data()
    #train_save_model(input_configs, output_config, Xt, yt)
    load_predict_model(Xv[:2000], yv[:2000])
