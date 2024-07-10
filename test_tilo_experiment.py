import pytest
from tilo_experiment import run_experiments
import math
import numpy
import pandas as pd

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
    max_iter = 12
    callback_iter = 3
    res = run_experiments(0, max_iter, optimizer="SGD", callback_interval=callback_iter, seed=47, num_data_points=2,
                         num_local_updates=1,
                         num_data_per_local_update_step=2,
                          model = "linear")

    assert isinstance(res, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(res) == math.ceil(max_iter / callback_iter) + 1
    expected_columns = ['mse', 'psnr', 'lpips', 'rpsnr', 'ssim', 'max_ssim', 'max_rpsnr', 'order', 'IIP-pixel',
                        'feat_mse', 'parameters', 'label_acc', 'loss', 'time', 'algorithm', 'model']

    assert [val >= 0 for val in res["ssim"]]
    assert list(
        res.columns) == expected_columns, f"DataFrame columns are not as expected. Expected: {expected_columns}, Got: {list(res.columns)}"


def test_optimizers():
    # make sure optimizers don't crash
    max_iter = 2
    callback_iter = 1

    def run_optim_experiment(optim, model="linear"):
        if optim == "KFAC":
            try:
                import kfac
            except ImportError:
                return
        run_experiments(0, max_iter, optimizer=optim, callback_interval=callback_iter, seed=47, num_data_points=2,
                        num_local_updates=1, num_data_per_local_update_step=2,
                          model = model)

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
