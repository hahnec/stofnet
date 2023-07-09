import wandb
import sys

api = wandb.Api()
runs = api.runs(path="hahnec/StofNet")

for run in runs:
    for artifact in run.logged_artifacts():
        if artifact.type == "data":
            artifact.download()
