import numpy as np
import pandas as pd
import logging, sys
import torch
import breaching
import sys
import argparse
import os


def configure_experiment(cfg, num_data_points, num_local_updates, num_data_per_local_update_step, optimizer,
                         optim_callback, max_iterations, model):
    cfg.case.data.partition = "random"
    cfg.case.user.user_idx = 1
    cfg.case.model = model
    cfg.case.user.provide_labels = True
    cfg.case.user.num_data_points = num_data_points
    cfg.case.user.num_local_updates = num_local_updates
    cfg.case.user.num_data_per_local_update_step = num_data_per_local_update_step
    cfg.attack.regularization.total_variation.scale = 1e-3
    cfg.case.user.optimizer = optimizer
    cfg.attack.optim.callback = optim_callback
    cfg.attack.optim.max_iterations = max_iterations
    return cfg


def run_experiments(gpu_index, max_iterations, name=None, optimizer="SGD",
                   seed=47, experiment_repetitions=1, callback_interval=100,
                   num_data_points=1, num_local_updates=1, num_data_per_local_update_step=1, model='resnet18'):
    """
    Run a gradient inversion attack experiment to test the ability to reconstruct images
    given a particular optimizer and configuration.

    Parameters:
    ----------
    gpu_index : int
        Index of the GPU to use. If no GPU is available, the CPU will be used.
    max_iterations : int
        Maximum number of iterations for the attack optimization process.
        Not sure if it's actually possible for it to end sooner.
    name : str
        Name of the experiment, used for naming output files and directories.
        If None, no files will be outputted.
    optimizer : str, optional
        Optimizer to use for the experiment (default is "SGD").
    experiment_repetitions : int, optional
        Number of times the experiment should be run (default is 1).
    seed : int, optional
    callback_interval : int, optional
        Interval of iterations at which the callback function is called to save metrics and images (default is 100).
    num_data_points : int, optional
    num_local_updates : int, optional
    num_data_per_local_update_step : int, optional

    Returns:
    -------
    user : object
        Needed to be able to create image of original and reconstructed data
        with the user.plot function.
    true_user_data : torch.Tensor
    reconstructed_user_data : torch.Tensor
    results_df : pandas.DataFrame

    Example:
    --------
    To run the experiment with specific parameters, execute the script from the command line:
    ```
    python tilo_experiment.py 0 1000 "SGD" "experiment_1"
    ```
    This will use GPU index 0, run for 1000 iterations, use the SGD optimizer, and save results in the "experiment_1" directory.
    """
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger()
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
    device = torch.device(f'cuda:{gpu_index}') if torch.cuda.is_available() else torch.device('cpu')
    if device == torch.device('cpu'):
        if name is None:
            print("using cpu")
        else:
            print(name + " using cpu")
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    cfg = configure_experiment(cfg, num_data_points, num_local_updates, num_data_per_local_update_step, optimizer,
                               callback_interval, max_iterations, model)

    def run_experiment(id):
        # Construct components
        user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
        attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
        #breaching.utils.overview(server, user, attacker)

        # Compute model update
        server_payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(server_payload)
        user.plot(true_user_data)

        # Reconstruct data
        results = []
        def calculate_metrics_callback(candidate, iteration, trial, labels):
            # save metrics data
            reconstructed_data = dict(data=candidate, labels=None)
            metrics = breaching.analysis.report(reconstructed_data, true_user_data, [server_payload],
                                                server.model, order_batch=True, compute_full_iip=False,
                                                cfg_case=cfg.case, setup=setup, verbose=False)
            metrics['iteration'] = iteration + 1
            results.append(metrics)

            if name is not None:
                user.plot(reconstructed_data)
                plt.savefig(name +"/inprogress_" + id + "/"+ name + "_" + id + "_iter"+str(iteration+1) + "_reconstruction.png")

        reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, callback=calculate_metrics_callback)

        return user, true_user_data, reconstructed_user_data, pd.DataFrame(results)

    if name is not None:
        folder = name + "/"
        make_folder(folder)

    dataframes = []

    for i in range(experiment_repetitions):
        experiment_id = "n" + str(i)
        if name is not None:
            make_folder(folder + "/inprogress_" + experiment_id)
        user, true_user_data, reconstructed_user_data, results_df = run_experiment(experiment_id)


        if name is not None:
            user.plot(true_user_data)
            plt.savefig(folder + name + "_"+experiment_id + '_ground_truth.png')
            user.plot(reconstructed_user_data)
            plt.savefig(folder + name + "_"+experiment_id + '_reconstruction.png')

            torch.save(true_user_data, folder + name + "_"+experiment_id + "_ground_truth.pt")
            torch.save(reconstructed_user_data, folder + name + "_"+experiment_id + "_reconstruction.pt")
            results_df.set_index('iteration').to_csv(folder + name + "_"+experiment_id + ".csv")

        results_df['experiment_id'] = experiment_id
        dataframes.append(results_df)

    all_results = pd.concat(dataframes).set_index(['experiment_id', 'iteration'])
    if name is not None:
        all_results.to_csv(folder + name  +".csv")
    return all_results


def summarize_dataframes(dataframes):
    # Ensure there is at least one DataFrame
    if len(dataframes) == 0:
        raise ValueError("At least one DataFrame is required")

    # Check all DataFrames have the same columns
    columns = dataframes[0].columns
    for df in dataframes:
        if not all(columns == df.columns):
            raise ValueError("All DataFrames must have the same columns")

    # Ensure all columns are numeric
    dataframes = [df.apply(pd.to_numeric, errors='coerce') for df in dataframes]

    # Concatenate the DataFrames along the rows
    combined_df = pd.concat(dataframes)

    # Calculate summary statistics
    avg_df = combined_df.groupby(combined_df.index).mean()
    std_df = combined_df.groupby(combined_df.index).std()
    q25_df = combined_df.groupby(combined_df.index).quantile(0.25)
    q50_df = combined_df.groupby(combined_df.index).quantile(0.50)
    q75_df = combined_df.groupby(combined_df.index).quantile(0.75)

    # Rename the columns
    avg_df.columns = [f'{col}_avg' for col in avg_df.columns]
    std_df.columns = [f'{col}_std' for col in std_df.columns]
    q25_df.columns = [f'{col}_25th' for col in q25_df.columns]
    q50_df.columns = [f'{col}_50th' for col in q50_df.columns]
    q75_df.columns = [f'{col}_75th' for col in q75_df.columns]

    # Combine all statistics into one DataFrame
    summary_df = pd.concat([avg_df, std_df, q25_df, q50_df, q75_df], axis=1)

    return summary_df

def make_folder(folder_name):
    # create folder to store experiment results
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Run experiment and save results to a CSV file.')
    parser.add_argument('gpu_index', type=int, help='GPU index to use.')
    parser.add_argument('max_iterations', type=int, help='Maximum number of iterations.')
    parser.add_argument('optimizer', type=str, help='Optimizer to use (e.g., SGD, Adam).')
    parser.add_argument('model', type=str, help='Optimizer to use (e.g., resnet18, linear, covnetsmall).')
    parser.add_argument('experiment_name', type=str, help='Output CSV file name.')
    parser.add_argument('--callback_interval', type=int, help='Interval at which the callback function is called', default=100)
    parser.add_argument('--batch_size', type=int, help='Number of images to use when calculating update',
                        default=1)
    parser.add_argument('--repetitions', type=int, help='Number of times to repeat experiment with different images',
                        default=1)
    args = parser.parse_args()


    run_experiments(
        name= args.experiment_name,
        gpu_index=args.gpu_index,
        max_iterations=args.max_iterations,
        optimizer=args.optimizer,
        model=args.model,
        experiment_repetitions=args.repetitions,
        callback_interval=args.callback_interval,
        num_data_points=args.batch_size, num_local_updates=1, num_data_per_local_update_step=args.batch_size
    )

    # folder = args.experiment_name + "/"
    # user.plot(true_user_data)
    # plt.savefig(folder + args.experiment_name +'_ground_truth.png')
    # user.plot(reconstructed_user_data)
    # plt.savefig(folder + args.experiment_name + '_reconstruction.png')
    #
    # torch.save(true_user_data, folder + args.experiment_name + "_ground_truth.pt")
    # torch.save(reconstructed_user_data, folder + args.experiment_name + "_reconstruction.pt")
    # results_df.to_csv(folder + args.experiment_name + ".csv")

