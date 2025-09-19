# This is a code made to validate the class done here but also to serve as a exemple to better understand
# how the classes here works and how you can use it in your case.

# this is the libraries I import to organize my tests
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# this is the classes you must import 
# import the data manager you created (see the TermoDataManager to understand how you should create this class)
from exemples.TermoDataManager import TermoDataManager
from models.FuzzyModel import FuzzyTSModel


def prepare_data(n_fuzzysets_list):
    """
    Prepare dataset and fuzzy variable configuration.

    Args:
        n_fuzzysets_list (list[int]): number of fuzzy sets for different groups of variables.

    Returns:
        input_configs (list[dict]): configuration for input fuzzy variables
        output_config (dict): configuration for output fuzzy variable
        Xt, yt, Xv, yv (np.ndarray): train/validation splits
    """
    # Dataset import
    path_to_data = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_files_name = [
        'Dados1_14a26_maio.txt',
        'Dados2_14a26_maio.txt',
        'Dados3_14a26_maio.txt',
        'Dados4_Power6a10_14a26_maio.txt'
    ]

    data_manager = TermoDataManager(path_to_data, data_files_name)
    data_in, data_out = data_manager.get_data_in_out(verbose=True)

    # For quick testing, use only the first 5% of data
    use_data_percentage = 0.05
    nb_of_samples = int(data_in.shape[0] * use_data_percentage)
    print(f"Using only {nb_of_samples} of {data_in.shape[0]} samples")
    data_in = data_in[:nb_of_samples]
    data_out = data_out[:nb_of_samples]

    # Variable ranges
    input_ranges = data_manager.get_dataframe_range(data_in)
    output_range = data_manager.get_dataframe_range(data_out)[0]

    # Add margin (10%) to ranges
    margin = 0.1
    input_ranges = [(a - margin * a, b + margin * b) for a, b in input_ranges]
    output_range = (output_range[0] - margin * output_range[0],
                    output_range[1] + margin * output_range[1])

    # Variable names
    input_names = data_in.columns
    output_name = data_out.columns[0]

    # Assign N for each variable
    n_list = [n_fuzzysets_list[0]] * 2 + [n_fuzzysets_list[1]] * 7

    # Split data
    Xt, yt, Xv, yv = data_manager.split_data(data_in, data_out, train_ratio=0.95)
    Xt, yt, Xv, yv = np.asarray(Xt), np.asarray(yt).ravel(), np.asarray(Xv), np.asarray(yv).ravel()

    # Build fuzzy model configs
    input_configs = [
        {"name": input_names[i], "N": n_list[i], "range": input_ranges[i]}
        for i in range(len(input_names))
    ]
    output_config = {"name": output_name, "N": n_fuzzysets_list[0], "range": output_range}

    return input_configs, output_config, Xt, yt, Xv, yv


def train_save_model(input_configs, output_config, Xt, yt, max_rules, test_description):
    """
    Train fuzzy model and save it.
    """
    model = FuzzyTSModel(input_configs=input_configs,
                         output_config=output_config,
                         max_rules=int(max_rules))

    # Modify log name to include test description
    new_log_name = model.log_name.split('_')[0] + test_description + "_".join(model.log_name.split('_')[1:])
    model.set_log_name(new_log_name)

    # Train model
    model.fit(Xt, yt)

    # Save trained model
    model_name = 'fuzzyTS_termo_exemple' + test_description
    path_to_save = os.path.join(os.getcwd(), 'exemples', 'saved_fuzzy_models')
    os.makedirs(path_to_save, exist_ok=True)
    model.save(os.path.join(path_to_save, model_name))


def load_predict_model(Xv, yv, n_fuzzysets_list, max_rules, test_description,
                        csv_path=os.path.join("exemples","results.csv")):
    """
    Load model, run prediction and save error plots.
    """
    # Load model
    path_to_load = os.path.join(os.getcwd(), 'exemples', 'saved_fuzzy_models',
                                'fuzzyTS_termo_exemple' + test_description)
    model = FuzzyTSModel.load(path_to_load)

    # Run prediction
    y_pred = model.predict(Xv)
    y_error = model.metrics.absolute_percentage_error(yv, y_pred)

    # --- Compute metrics ---
    mae = model.metrics.mean_absolute_error(yv, y_pred)
    mape = model.metrics.mean_absolute_percentage_error(yv, y_pred)
    rmse = model.metrics.root_mean_squared_error(yv, y_pred)
    r2 = model.metrics.r2_score(yv, y_pred)

    print(f"Metrics for {test_description}:")
    print(f"MAE={mae:.4f}, MAPE={mape:.2f}%, RMSE={rmse:.4f}, R2={r2:.4f}")

    # --- Save metrics to CSV ---
    header = ["model_name", "MAE", "MAPE", "RMSE", "R2"]
    row = [test_description, mae, mape, rmse, r2]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:  # write header only once
            writer.writerow(header)
        writer.writerow(row)

    # Plot results
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(yv, color='red', label="Ground Truth")
    ax[0].plot(y_pred, linestyle='--', color='blue', label="Prediction")
    ax[0].set_ylim([0, 1])
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(y_error, linestyle='--', color='black', label="Absolute Percent Error")
    ax[1].set_ylim([0, 100])
    ax[1].set_ylabel("Error (%)")
    ax[1].grid(True)
    # Save plot
    img_path = os.path.join(os.getcwd(), 'exemples', 'error_figs')
    os.makedirs(img_path, exist_ok=True)
    img_name = f"fuzzy_error_FS{n_fuzzysets_list[0]}_{n_fuzzysets_list[1]}_R{max_rules}"
    plt.savefig(os.path.join(img_path, img_name + '.png'))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and evaluate fuzzy models with variable configs.")
    parser.add_argument("--max_rules", type=int, required=True,
                        help="Maximum number of fuzzy rules.")
    parser.add_argument("--n_fuzzysets", nargs=2, type=int, required=True,
                        help="Two integers defining fuzzy sets for groups of variables, e.g. --n_fuzzysets 10 5")
    parser.add_argument("--predict_mode", action="store_true",
                        help="If set, skip training and only run prediction with an existing model.")
    
    args = parser.parse_args()

    n_fuzzysets_list = args.n_fuzzysets
    max_rules = args.max_rules
    test_description = f"_FS_{n_fuzzysets_list[0]}_{n_fuzzysets_list[1]}_R_{max_rules}"

    # Prepare data
    input_configs, output_config, Xt, yt, Xv, yv = prepare_data(n_fuzzysets_list)

    # Train and save model (if predict_mode is not active)
    if not args.predict_mode:
        train_save_model(input_configs, output_config, Xt, yt, max_rules, test_description)

    # Load model and run prediction
    load_predict_model(Xv, yv, n_fuzzysets_list, max_rules, test_description)


if __name__ == '__main__':
    main()
