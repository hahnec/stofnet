import numpy as np
import wandb
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def wb_img_upload(fig, log_key='img'):

    fig.canvas.draw()
    width, height = [int(round(hw)) for hw in fig.get_size_inches() * fig.get_dpi()]
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    wandb.log({log_key: wandb.Image(img)})
    plt.close(fig)


def plot_channel_overview(frame, corresponding_toas, echoes=None, max_val=None, magnify_adjacent=False, magnify_from=None, figsize=(13, 7)):

    max_val = float(np.quantile(abs(frame), .99)) if max_val is None else max_val
    
    ch_num = frame.shape[-2]
    ch_min = (ch_num-4)//2 if magnify_from is None else magnify_from
    ch_min = ch_min if magnify_adjacent else 0
    ch_max = ch_min+4 if magnify_adjacent else ch_num
    colors = ['red', 'green', 'orange', 'pink', 'gray', 'brown', 'violet', 'magenta', 'cyan', 'yellow']
    fig, axs = plt.subplots(nrows=4 if magnify_adjacent else ch_num, ncols=1, figsize=figsize)
    if not magnify_adjacent: axs = [axs] # fix for index error when nrows=1, ncols=1
    for j, i in enumerate(range(ch_min, ch_max, 1)):
        axs[j].plot(frame[i, :])
        axs[j].plot(abs(hilbert(frame[i, :])), c='gray')
        max_echoes = max(corresponding_toas.shape[-1], max([len(el) for el in echoes]))
        for c in range(max_echoes):
            axs[j].plot([corresponding_toas[i, c],]*2, [.8*max_val, -.8*max_val], c=colors[c%len(colors)]) if i < corresponding_toas.shape[-2] and c < corresponding_toas.shape[-1] else None
            axs[j].plot([echoes[i][c],]*2, [max_val, -max_val], c='black', linestyle='dashed') if echoes is not None and i < len(echoes) and c < len(echoes[i]) else None
            axs[j].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    plt.tight_layout()

    return fig
