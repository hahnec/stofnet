import wandb

runs = wandb.Api().runs("StofNet")

# Sort the runs by creation time (most recent first)
sorted_runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]) if run.state == 'finished' else 0, reverse=True)

# retrieve a specific number of most recent runs
num_recent_runs = 4
recent_runs = sorted_runs[:num_recent_runs]

ndigits = 3
metric_runs = []
for metric_run in recent_runs:
    model_name = metric_run.summary['model_name'] if 'model_name' in metric_run.summary.keys() else None
    total_dist_mean = metric_run.summary['total_distance_mean']
    total_dist_std = metric_run.summary['total_distance_std']
    total_time = metric_run.summary['total_inference_time'] * 1e3
    total_params = metric_run.summary['total_parameters']
    total_jaccard = metric_run.summary['total_jaccard']
    # replace None with '-'

    model_name, total_dist_mean, total_dist_std, total_time, total_params, total_jaccard = [round(el, ndigits) if isinstance(el, (float, int)) else el for el in [model_name, total_dist_mean, total_dist_std, total_time, total_params, total_jaccard]]
    row_list = [str(model_name), str(total_params), str(total_dist_mean)+' $\pm$ '+str(total_dist_std), str(total_jaccard), str(total_time)]
    metric_runs.append(row_list)

with open("metrics_table.tex", "w") as f:
    # Write the LaTeX table header
    f.write("\\begin{tabularx}{\linewidth}{ \n \
        |>{\centering\\arraybackslash}p{5em} \n \
        |>{\centering\\arraybackslash}p{3em} \n \
        |>{\centering\\arraybackslash}p{6em} \n \
        |>{\centering\\arraybackslash}p{3em} \n \
        |>{\centering\\arraybackslash}p{3em} \n \
        |}\n")
    f.write("\\cline{1-5}\n")
    f.write("Model & Weights [\#] & RMSE [Sample] & Jaccard [\%] & Time [ms] \\\\\n")
    f.write("\\cline{1-5}\n")

    # Iterate over the runs and extract metrics
    for k, row_list in enumerate(metric_runs):
        
        # replace model entry
        row_list[0] = ['Gradient \cite{Hahne:22}', 'Zonzini \cite{Zonzini:2022}', 'SincNet \cite{ravanelli2018speaker}', 'Ours'][k]

        # replace None entries
        row_list = [col.replace('None', 'n.a.') for col in row_list]

        # Write a row for each metric in the LaTeX table
        f.write(" & ".join(str(metric) for metric in row_list))
        f.write(" \\\\\n")

    # Write the LaTeX table footer
    f.write("\\cline{1-5}\n")
    f.write("\\end{tabularx}")

artifact = None
if artifact:
    artifact = metric_run.use_artifact('artifact_name:version', type='artifact_type')
    artifact_dir = artifact.download()
