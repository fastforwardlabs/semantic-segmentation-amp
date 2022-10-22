from src.train import train_model
from src.model_utils import (
    tversky,
    tversky_loss,
    tversky_axis,
    tversky_loss_axis,
)

IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOSSES = {
    "tversky_loss": tversky_loss,
    "tversky_loss_axis": tversky_loss_axis,
}
METRICS = {
    "tversky": tversky,
    "tversky_axis": tversky_axis,
}

kwargs = {
    "n_epochs": 100,
    "learning_rate": 0.01,
    "n_channels_bottleneck": 512,
    "loss_fn": "tversky_loss_axis",
    "sample_weights": True,
    "custom_log_dir": None,
    "small_sample": True,
    "img_shape": IMG_SHAPE,
    "batch_size": BATCH_SIZE,
    "annotations_path": ANNOTATIONS_PATH,
    "train_img_path": TRAIN_IMG_PATH,
    "losses": LOSSES,
    "metrics": METRICS,
    "sample_weight_strategy": "ens",
    "sample_weight_ens_beta": 0.99,
}


def get_kwargs_copy():
    return kwargs.copy()


# run all versions of ENS
sample_weight_strategy = "ens"
sample_weights = True

# for sample_weight_ens_beta in [0.9, 0.99, 0.999, 0.9999]:
for sample_weight_ens_beta in [0.999, 0.9999]:
    run_kwargs = get_kwargs_copy()
    run_kwargs.update(
        {
            "sample_weight_strategy": sample_weight_strategy,
            "sample_weight_ens_beta": sample_weight_ens_beta,
            "sample_weights": sample_weights,
            "custom_log_dir": f"logs2/experiment1-sample_weights_{sample_weights}-strategy_{sample_weight_strategy}-beta_{sample_weight_ens_beta}",
        }
    )

    train_model(**run_kwargs)

# run IP
sample_weight_strategy = "ip"
sample_weight_ens_beta = "NA"
sample_weights = True
run_kwargs = get_kwargs_copy()
run_kwargs.update(
    {
        "sample_weight_strategy": sample_weight_strategy,
        "sample_weights": sample_weights,
        "custom_log_dir": f"logs2/experiment1-sample_weights_{sample_weights}-strategy_{sample_weight_strategy}-beta_{sample_weight_ens_beta}",
    }
)
train_model(**run_kwargs)

# run no sample weighting
sample_weight_strategy = "ip"
sample_weight_ens_beta = "NA"
sample_weights = False
run_kwargs = get_kwargs_copy()
run_kwargs.update(
    {
        "sample_weight_strategy": sample_weight_strategy,
        "sample_weights": sample_weights,
        "custom_log_dir": f"logs2/experiment1-sample_weights_{sample_weights}-strategy_{'NA'}-beta_{sample_weight_ens_beta}",
    }
)
train_model(**run_kwargs)
