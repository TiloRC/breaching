import pytest
from tilo_experiment import run_experiments
import math
import numpy
import pandas as pd
from breaching import get_config
from breaching import get_config
from breaching.cases import construct_dataloader, construct_case
import torch
from collections import Counter

#disable breakpoints
import pdb
def noop():
    pass
pdb.set_trace = noop


# set up config
def make_config():
    cfg = get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
    cfg.case.data.partition = "random"
    cfg.case.user.user_idx = 1
    cfg.case.model = "linear"
    cfg.case.user.provide_labels = True
    cfg.case.user.num_data_points = 1
    cfg.case.user.num_local_updates = 1
    cfg.case.user.num_data_per_local_update_step =1
    cfg.attack.regularization.total_variation.scale = 1e-3
    cfg.case.user.optimizer = 'SGD'
    cfg.attack.optim.callback = 1000
    cfg.attack.optim.max_iterations = 1
    cfg.case.data.classes_per_batch = 1
    return cfg

def test_cuda():
    import platform

    if platform.system() == "Linux":
        from torch import cuda
        assert cuda.is_available()


def test_dependencies():
    from pytorch_wavelets import DTCWTForward

def test_run_experiments():
    numpy.array([1, 2])
    print(numpy.__version__)

    cfg = make_config()
    cfg.attack.optim.max_iterations = 12
    cfg.attack.optim.callback = 3
    res = run_experiments(cfg, 0)

    assert isinstance(res, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(res) == math.ceil(cfg.attack.optim.max_iterations / cfg.attack.optim.callback) + 1
    expected_columns = ['mse', 'psnr', 'lpips', 'rpsnr', 'ssim', 'max_ssim', 'max_rpsnr', 'order', 'IIP-pixel', 'feat_mse', 'parameters', 'label_acc', 'loss', 'time']


    assert [val >= 0 for val in res["ssim"]]
    # assert list(
    #     res.columns) == expected_columns, f"DataFrame columns are not as expected. Expected: {expected_columns}, Got: {list(res.columns)}"


def test_optimizers():
    cfg = make_config()
    # make sure optimizers don't crash
    cfg.attack.optim.max_iterations = 2
    cfg.attack.optim.callback = 1

    def run_optim_experiment(optim, model="linear"):
        if optim == "KFAC":
            try:
                import kfac
            except ImportError:
                return
        cfg.case.model = model
        cfg.case.user.optimizer = optim

        run_experiments(cfg, 0)

    run_optim_experiment("SGD")
    run_optim_experiment("KFAC")
    run_optim_experiment("Adam")
    run_optim_experiment("Adagrad")
    run_optim_experiment("RMSprop")
    run_optim_experiment("SGD_with_momentum")
    run_optim_experiment("LBFGS")
    run_optim_experiment("Random")
    run_optim_experiment("Random", "convnetsmall")

    with pytest.raises(ValueError, match="Unknown optimizer: UnknownOptimizer"):
        run_optim_experiment("UnknownOptimizer")

def test_batch_heterogeneity():
    def get_data(cfg):
        device = torch.device('cpu')
        setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
        loader = construct_dataloader(cfg.case.data, cfg.case.impl)
        user, server, model, loss_fn = construct_case(cfg.case, setup)
        server_payload = server.distribute_payload()
        update = list(user.compute_local_updates(server_payload))
        assert len(update) == 1
        shared_data, true_user_data, metrics = update[0]
        return true_user_data

    # test experiments without batch hetergeneity work normally
    cfg = make_config()
    cfg.case.data.classes_per_batch = None
    cfg.case.data.batch_size = 8
    cfg.case.user.num_data_points = 8

    res = run_experiments(cfg, 0)
    assert isinstance(res, pd.DataFrame)

    labels1 = get_data(cfg)['labels'].tolist()
    labels2 = get_data(cfg)['labels'].tolist()
    labels3 = get_data(cfg)['labels'].tolist()
    count1 = len(set(labels1))
    count2 = len(set(labels2))
    count3 = len(set(labels3))
    # this will sometimes fail even if everything is working correctly
    assert (count1 != count2) or (count2 != count3)


    # batch_size == 4
    cfg = get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
    cfg.case.data.classes_per_batch = 2
    cfg.case.data.batch_size = 4
    cfg.case.user.num_data_points = 4
    max_observations_per_class = math.ceil(cfg.case.data.batch_size / cfg.case.data.classes_per_batch)
    true_user_data = get_data(cfg)

    labels = true_user_data['labels'].tolist()
    assert len(labels) == cfg.case.data.batch_size
    assert len(set(labels)) == cfg.case.data.classes_per_batch
    assert all(val <= max_observations_per_class for val in Counter(labels).values())


    # batch_size == 8
    cfg = get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
    cfg.case.data.classes_per_batch = 2
    cfg.case.data.batch_size = 8
    cfg.case.user.num_data_points = 8
    max_observations_per_class = math.ceil(cfg.case.data.batch_size / cfg.case.data.classes_per_batch)
    true_user_data = get_data(cfg)

    labels = true_user_data['labels'].tolist()
    assert len(labels) == cfg.case.data.batch_size
    assert len(set(labels)) == cfg.case.data.classes_per_batch
    assert all(val <= max_observations_per_class for val in Counter(labels).values())


def test_attack_during_training():
    cfg = make_config()
    cfg.case.user.run_attacks_during_training = True
    cfg.case.user.num_local_updates = 2
    # ensure one row per attack
    cfg.attack.optim.max_iterations = 1

    res = run_experiments(cfg, 0)




    assert "training accuracy" in list(res.columns)
    assert "training loss" in list(res.columns)
    assert res.shape[0] == 2 # num rows should be two: one for each attack
    assert list(res['ssim'])[0] != list(res['ssim'])[1] # each attack should have different reconstructions

