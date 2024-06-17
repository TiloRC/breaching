import pytest
from tilo_experiment import run_experiment
import math
import numpy
import pandas as pd


def test_run_experiment():
    numpy.array([1, 2])
    print(numpy.__version__)
    max_iter = 12
    callback_iter = 3
    res = run_experiment(2, max_iter, optimizer="SGD", optim_callback=callback_iter, seed=47, num_data_points=2,
                         num_local_updates=1,
                         num_data_per_local_update_step=2)

    assert isinstance(res, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(res) == math.ceil(max_iter / callback_iter) + 1
    expected_columns = ['mse', 'psnr', 'lpips', 'rpsnr', 'ssim', 'max_ssim', 'max_rpsnr', 'order', 'IIP-pixel',
                        'feat_mse', 'parameters', 'label_acc']
    assert list(
        res.columns) == expected_columns, f"DataFrame columns are not as expected. Expected: {expected_columns}, Got: {list(res.columns)}"


def test_optimizers():
    # make sure optmizers don't crash
    max_iter = 1
    callback_iter = 1
    for optim in ["SGD", "KFAC", "Adam", "Adagrad", "RMSprop", "SGD_with_momentum", "Random"]:
        if optim == "KFAC":
            try:
                import kfac
            except ImportError:
                continue
        run_experiment(2, max_iter, optimizer=optim, optim_callback=callback_iter, seed=47, num_data_points=2,
                         num_local_updates=1,
                         num_data_per_local_update_step=2)

    with pytest.raises(ValueError, match="Unknown optimizer: UnknownOptimizer"):
        run_experiment(2, max_iter, optimizer="UnknownOptimizer", optim_callback=callback_iter, seed=47,
                       num_data_points=2,
                       num_local_updates=1, num_data_per_local_update_step=2)

