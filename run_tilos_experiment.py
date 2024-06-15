import numpy as np
import logging, sys
import torch
import breaching


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


def run_experiment(gpu_index, max_iterations, optimizer="SGD", optim_callback=1000,
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
                               optim_callback, max_iterations)

    # Construct components
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    #breaching.utils.overview(server, user, attacker)

    # Train model
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)
    user.plot(true_user_data)

    # Reconstruct data
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

    reconstructed_user_data["labels"] = None # prevents weird bug with breaching.analysis.report
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                        server.model, order_batch=True, compute_full_iip=False,
                                        cfg_case=cfg.case, setup=setup)
    return metrics


if __name__ == "__main__":
    run_experiment(2,1000, optimizer="SGD", optim_callback=100, seed=47, num_data_points=2, num_local_updates=1,
                   num_data_per_local_update_step=2)

    from breaching.analysis import load_reconstruction

    load_reconstruction(0, 0, "cpu")