import numpy as np
import pandas as pd
import logging, sys
import torch
import breaching
import sys
import argparse
import os


def configure_experiment(cfg, num_data_points, num_local_updates, num_data_per_local_update_step, optimizer,
                         optim_callback, max_iterations):
    cfg.case.data.partition = "random"
    cfg.case.user.user_idx = 1
    cfg.case.model = 'resnet18'
    cfg.case.user.provide_labels = True
    cfg.case.user.num_data_points = num_data_points
    cfg.case.user.num_local_updates = num_local_updates
    cfg.case.user.num_data_per_local_update_step = num_data_per_local_update_step
    cfg.attack.regularization.total_variation.scale = 1e-3
    cfg.case.user.optimizer = optimizer
    cfg.attack.optim.callback = optim_callback
    cfg.attack.optim.max_iterations = max_iterations
    return cfg


def run_experiment(gpu_index, max_iterations, name, optimizer="SGD", callback_interval=100,
                   num_data_points=2, num_local_updates=1, num_data_per_local_update_step=2,
                   seed=47):
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger()
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
    device = torch.device(f'cuda:{gpu_index}') if torch.cuda.is_available() else torch.device('cpu')
    if device == torch.device('cpu'):
        print("Using cpu")
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    cfg = configure_experiment(cfg, num_data_points, num_local_updates, num_data_per_local_update_step, optimizer,
                               callback_interval, max_iterations)

    # Construct components
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    #breaching.utils.overview(server, user, attacker)

    # Train model
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

        # save reconstruction image
        #if (iteration+1) % (callback_interval*2) == 0:
        user.plot(reconstructed_data)
        plt.savefig(name +"/inprogress/"+ name + "_iter"+str(iteration+1) + "_reconstruction.png")

    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, callback=calculate_metrics_callback)

    return user, true_user_data, reconstructed_user_data, pd.DataFrame(results).set_index('iteration')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Run experiment and save results to a CSV file.')
    parser.add_argument('gpu_index', type=int, help='GPU index to use.')
    parser.add_argument('max_iterations', type=int, help='Maximum number of iterations.')
    parser.add_argument('optimizer', type=str, help='Optimizer to use (e.g., SGD, Adam).')
    parser.add_argument('experiment_name', type=str, help='Output CSV file name.')
    parser.add_argument('--callback_interval', type=int, help='Interval at which the callback function is called', default=100)
    parser.add_argument('--batch_size', type=int, help='Number of images to use when calculating update',
                        default=1)
    args = parser.parse_args()

    # create folder to store experiment results
    try:
        os.makedirs(args.experiment_name)
    except FileExistsError:
        pass
    os.makedirs(args.experiment_name + "/inprogress") # folder for in progress images

    user, true_user_data, reconstructed_user_data, results_df = run_experiment(
        name= args.experiment_name,
        gpu_index=args.gpu_index,
        max_iterations=args.max_iterations,
        optimizer=args.optimizer,
        callback_interval=args.callback_interval,
        num_data_points=args.batch_size, num_local_updates=1, num_data_per_local_update_step=args.batch_size
    )

    folder = args.experiment_name + "/"
    user.plot(true_user_data)
    plt.savefig(folder + args.experiment_name +'_ground_truth.png')
    user.plot(reconstructed_user_data)
    plt.savefig(folder + args.experiment_name + '_reconstruction.png')

    torch.save(true_user_data, folder + args.experiment_name + "_ground_truth.pt")
    torch.save(reconstructed_user_data, folder + args.experiment_name + "_reconstruction.pt")
    results_df.to_csv(folder + args.experiment_name + ".csv")

