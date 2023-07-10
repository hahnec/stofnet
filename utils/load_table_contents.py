import wandb
import sys
from pathlib import Path
import json
import numpy as np
import shutil

api = wandb.Api()
runs = api.runs("StofNet")

# filter group runs
group_name = sys.argv[1] if len(sys.argv) > 1 else 'chirp_array'
runs = [run for run in runs if run.group == group_name]

# Sort the runs by creation time (most recent first)
sorted_runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]) if run.state == 'finished' else 0, reverse=True)

# retrieve a specific number of most recent runs
num_recent_runs = 5
recent_runs = sorted_runs[:num_recent_runs]

# artifact handling
load_artifact_opt = False
if load_artifact_opt and Path('./artifacts').exists(): shutil.rmtree('./artifacts')

ndigits = 3
metric_runs = []
toas, model_names = [], []
for metric_run in recent_runs:
    model_name = metric_run.summary['model_name'] if 'model_name' in metric_run.summary.keys() else None
    total_dist_mean = metric_run.summary['total_distance_mean']
    total_dist_std = metric_run.summary['total_distance_std']
    total_time = metric_run.summary['total_inference_time'] * 1e3
    total_params = metric_run.summary['total_parameters'] / 1e3
    total_jaccard = metric_run.summary['total_jaccard']
    # replace None with '-'

    fmt = '.'+str(ndigits)+'f'
    model_name, total_dist_avg, total_dist_std, total_time, total_params, total_jaccard = [format(round(el, ndigits), fmt) if isinstance(el, (float, int)) else el for el in [model_name, total_dist_mean, total_dist_std, total_time, total_params, total_jaccard]]
    row_list = [str(model_name), str(total_dist_avg)+' $\pm$ '+str(total_dist_std), str(total_jaccard), str(total_params).split('.')[0], str(total_time)]
    metric_runs.append(row_list)
    
    # download artifacts
    for artifact in metric_run.logged_artifacts():
        if artifact.type == "data":
            artifact_path = Path('./artifacts') / model_name / artifact.name.split(':')[0]
            if load_artifact_opt: artifact.download(artifact_path)
    
    # load artifacts
    stack = []
    for name in ['toa', 'gt']:
        with open(artifact_path / (name+'.table.json'), 'r') as json_file:
            json_dict = json.load(json_file)
            stack.append(np.array(json_dict['data']))

    g=[np.load(f) for f in (artifact_path/'media'/'serialized_data').iterdir() if str(f).endswith('npz')][0]
    frame = g['Column0'].squeeze()

    toa, gt = stack
    toas.append(toa)
    model_names.append(model_name)

# write table
with open("metrics_table.tex", "w") as f:
    # Write the LaTeX table header
    f.write("\\begin{tabularx}{\linewidth}{ \n \
        >{\\raggedright\\arraybackslash}p{4.9em} %| \n \
        >{\centering\\arraybackslash}p{5.76em} %| \n \
        S[table-format=2.3] %| \n \
        >{\\raggedleft\\arraybackslash}p{2.9em} %| \n \
        >{\centering\\arraybackslash}p{2em} %| \n \
        }\n")
    f.write("\\toprule\n")
    f.write("\centering Model & {RMSE [Sample]} & {Jaccard~[\%]}& \centering Weights [k\#]  & Time [ms] \\\\\n")
    f.write("\\midrule\n")

    # Iterate over the runs and extract metrics
    for k, row_list in enumerate(metric_runs):
        
        # replace model entry
        row_list[0] = ['Gradient~\cite{Hahne:22}', 'Zonzini~\cite{Zonzini:2022}', 'Kuleshov~\cite{kuleshov2017audio}', 'SincNet~\cite{ravanelli2018speaker}', 'Ours'][k]

        # replace None entries
        row_list = [col.replace('None', 'n.a.') for col in row_list]

        # Write a row for each metric in the LaTeX table
        f.write(" & ".join(str(metric) for metric in row_list))
        f.write(" \\\\\n")

    # Write the LaTeX table footer
    f.write("\\bottomrule\n")
    f.write("\\end{tabularx}")

# create plot
from stofnet.utils.plot_frame import stofnet_plot
stofnet_plot(frame, toa_list=[gt]+toas, toa_labels=['Ground Truth']+model_names)
