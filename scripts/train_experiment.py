import json
import cmlapi

# instantiate CML API client
client = cmlapi.default_client()

# get project id
projects = client.list_projects(search_filter=json.dumps({"name": "FF25_AMP_dev"}))
PROJECT_ID = projects.projects[0].id

# get runtime id
py39_gpu_runtimes = client.list_runtimes(
    search_filter=json.dumps(
        {
            "kernel": "Python 3.9",
            "edition": "Nvidia GPU",
            "editor": "Workbench",
            "version": "2022.04",
            "image_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-cuda:2022.04.1-b6",
        }
    )
)
RUNTIME_ID = py39_gpu_runtimes.runtimes[0].image_identifier


# configure experiments
N_EPOCHS = 75
LEARNING_RATE = 0.01
N_CHANNELS_BOTTLENECK = 512
LOSS_FN = "tversky_loss_axis"

experiments = [
    {
        "sample_weight_strategy": "ens",
        "sample_weight_ens_beta": 0.99,
        "sample_weights": False,
    },
    {
        "sample_weight_strategy": "ens",
        "sample_weight_ens_beta": 0.99,
        "sample_weights": True,
    },
    {
        "sample_weight_strategy": "ens",
        "sample_weight_ens_beta": 0.999,
        "sample_weights": True,
    },
    {
        "sample_weight_strategy": "ens",
        "sample_weight_ens_beta": 0.9999,
        "sample_weights": True,
    },
    {
        "sample_weight_strategy": "ip",
        "sample_weight_ens_beta": 0,
        "sample_weights": True,
    },
]


for i, experiment in enumerate(experiments):

    ARGS = f"--n_epochs {N_EPOCHS} --lr {LEARNING_RATE} --n_channels {N_CHANNELS_BOTTLENECK} --loss_fn {LOSS_FN} --sample_weight_strategy {experiment['sample_weight_strategy']} --sample_weight_ens_beta {experiment['sample_weight_ens_beta']} --small_sample {'--sample_weights' if experiment['sample_weights'] else ''}"

    # create job
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Sample Weights Experiment {i}",
        script="scripts/run_train.py",
        runtime_identifier=RUNTIME_ID,
        cpu=4.0,
        memory=8.0,
        nvidia_gpu=1,
        arguments=ARGS,
        parent_job_id=None if i == 0 else job.id,
    )

    # Create this job within the project specified by the project_id parameter.
    job = client.create_job(job_body, PROJECT_ID)
