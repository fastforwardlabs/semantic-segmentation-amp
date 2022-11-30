import json
import cmlapi


def create_train_jobs():
    print("Creating experimental training jobs.")

    # instantiate CML API client
    client = cmlapi.default_client()

    # get project id
    projects = client.list_projects(
        search_filter=json.dumps({"name": "CML_AMP_Defect_Detection"})
    )
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
    N_EPOCHS = 200
    LEARNING_RATE = 0.01
    N_CHANNELS_BOTTLENECK = 352

    experiments = [
        {
            "alias": "baseline",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": False,
            "sample_weight_strategy": "NA",
            "undersample_train_set": False,
            "oversample_train_set": False,
        },
        {
            "alias": "undersample",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": False,
            "sample_weight_strategy": "NA",
            "undersample_train_set": True,
            "oversample_train_set": False,
        },
        {
            "alias": "oversample",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": False,
            "sample_weight_strategy": "NA",
            "undersample_train_set": False,
            "oversample_train_set": True,
        },
        {
            "alias": "sample_weighting_ip",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": True,
            "sample_weight_strategy": "ip",
            "undersample_train_set": False,
            "oversample_train_set": False,
        },
        {
            "alias": "sample_weighting_ens_0.99",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": True,
            "sample_weight_strategy": "ens",
            "sample_weight_ens_beta": 0.99,
            "undersample_train_set": False,
            "oversample_train_set": False,
        },
        {
            "alias": "sample_weighting_ens_0.999",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": True,
            "sample_weight_strategy": "ens",
            "sample_weight_ens_beta": 0.999,
            "undersample_train_set": False,
            "oversample_train_set": False,
        },
        {
            "alias": "sample_weighting_ens_0.9999",
            "loss_fn": "categorical_crossentropy",
            "sample_weights": True,
            "sample_weight_strategy": "ens",
            "sample_weight_ens_beta": 0.9999,
            "undersample_train_set": False,
            "oversample_train_set": False,
        },
    ]

    for i, experiment in enumerate(experiments):

        ARGS = f"--n_epochs {N_EPOCHS} \
                --lr {LEARNING_RATE} \
                --n_channels {N_CHANNELS_BOTTLENECK} \
                --loss_fn {experiment['loss_fn']} \
                {'--sample_weights' if experiment['sample_weights'] else ''} \
                {'--sample_weight_strategy '+experiment['sample_weight_strategy'] if experiment['sample_weights'] else ''} \
                {'--sample_weight_ens_beta '+str(experiment['sample_weight_ens_beta']) if (experiment['sample_weights']) & (experiment['sample_weight_strategy']=='ens') else ''} \
                {'--oversample_train_set' if experiment['oversample_train_set'] else ''} \
                {'--undersample_train_set' if experiment['undersample_train_set'] else ''}"

        # create job
        job_body = cmlapi.CreateJobRequest(
            project_id=PROJECT_ID,
            name=f"Imbalanced Data Experiment - {experiment['alias'].upper()}",
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

        print(
            "Finished creating experimental training jobs. Run the BASELINE experiment \
             first as all other jobs are downstream dependencies of this."
        )


if __name__ == "__main__":
    create_train_jobs()
