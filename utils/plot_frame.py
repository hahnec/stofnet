import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from pathlib import Path
import numpy as np


def stofnet_plot(channel_data, toa_list, toa_labels, xs1=0, xs2=-1, xs3=None, xs4=None, x=None):

    max_val = max(abs(channel_data))
    x = np.arange(len(channel_data)) if x is None else x

    width = 60
    xs3 = int(toa_list[0])-width//2 if xs3 is None else xs3
    xs4 = int(toa_list[0])+width//2 if xs4 is None else xs4

    colors = ['#0051a2', 'darkgreen', '#ffd44f', '#fd271f', '#93003a', '#808080', '#601090']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'x', '+', '.']
    heights = [.45, .3, .15, .075, -.075, -.15]
    lwidths = [.5, 1.75, 1, 1, .75, 1.5]

    # Create main container
    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(bottom = 0., left = 0, top = 1., right = 1)

    # Create upper axes
    sub1 = fig.add_subplot(1,3,(1,2))
    (l1,) = sub1.plot(x[xs1:xs2], channel_data[xs1:xs2], linestyle='solid', linewidth=lwidths[0], color='k', label="${y}$ input signal")
    sub1.set_xlim(x[xs1], x[xs2])
    sub1.set_ylim(-max_val, max_val)
    sub1.set_ylabel(r'Amplitude [a.u.]', fontsize=24, labelpad = 15)
    sub1.tick_params(axis='both', which='major', labelsize=15)
    sub1.tick_params(axis='both', which='minor', labelsize=13)
    sub1.set_xlabel(r'Time $\mathbf{x}$ [sample]', fontsize=24, labelpad = 15)

    # Create upper axes
    sub2 = fig.add_subplot(1,3,3)
    rect_center_x = (x[xs3]+x[xs4])/2
    rect_height = 0.39
    sub2.plot(x[xs3:xs4], channel_data[xs3:xs4], linestyle='solid', linewidth=lwidths[0]*2, color='k')
    sub2.set_xlim(x[xs3], x[xs4])
    sub2.set_ylim(-rect_height/2, rect_height/2)
    #sub2.set_ylabel(r'Amplitude [a.u.]', fontsize=24, labelpad = 15)
    sub2.tick_params(axis='both', which='major', labelsize=15)
    sub2.tick_params(axis='both', which='minor', labelsize=13)
    sub2.set_xlabel(r'Time $\mathbf{x}$ [sample]', fontsize=24, labelpad = 15)

    # ground truth
    (l2,) = sub1.plot([toa_list[0].squeeze(),]*2, [max_val, -max_val], c='red', linestyle='dashed', label=toa_labels[0])
    sub2.plot([toa_list[0].squeeze(),]*2, [max_val, -max_val], c='red', linestyle='dashed')

    # Time-of-Arrivals
    for toa, label, c, marker, height in zip(toa_list[1:], toa_labels[1:], colors[:len(toa_list)-1], markers[:len(toa_list)-1], heights):
        (l2,) = sub1.plot(toa.squeeze(), height, c=c, label=label.capitalize(), linestyle='', marker=marker, markersize=12)
        sub2.plot(toa.squeeze(), height, c=c, linestyle='', marker=marker, markersize=12)

    # Create left side of Connection patch for first axes
    con1 = ConnectionPatch(xyA=(x[xs3], sub2.get_ylim()[1]), coordsA=sub2.transData, 
                        xyB=(x[xs3], rect_height/2), coordsB=sub1.transData, color='k', linewidth=.7)
    # Add left side to the figure
    fig.add_artist(con1)

    # Create right side of Connection patch for first axes
    con2 = ConnectionPatch(xyA=(x[xs3], sub2.get_ylim()[0]), coordsA=sub2.transData, 
                        xyB=(x[xs3], -rect_height/2), coordsB=sub1.transData, color='k', linewidth=.7)
    # Add right side to the figure
    fig.add_artist(con2)

    rect = Rectangle((rect_center_x-(x[xs4]-x[xs3])/2, -rect_height/2), width=(x[xs4]-x[xs3]), height=rect_height, fill=False, color='k', linewidth=.7)
    sub1.add_artist(rect)

    handles, labels = sub1.get_legend_handles_labels()
    fig.legend([l1, l2], labels=labels, fontsize=21.5, fancybox=True, framealpha=1, ncol=2, bbox_to_anchor=(0.001, 0.5, 0.5, 0.5))

    # Save figure with nice margin
    plt.tight_layout(rect=(0,0,1,.89))
    plt.savefig(Path.cwd() / 'plot.svg', transparent=False, dpi=300, pad_inches=.0)
    plt.savefig(Path.cwd() / 'plot.eps', transparent=False, dpi=300, pad_inches=.0)
    plt.show()


if __name__ == '__main__':

    # Generate some example data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x, y, color='blue')

    # Enlarge a segment of the plot
    axins = ax.inset_axes([0.6, 0.1, 0.3, 0.4])
    axins.plot(x[2:4], y[2:4], color='red')
    ax.indicate_inset_zoom(axins, edgecolor='gray')

    # Add connection lines at the side
    ax.axvline(x=3, ymin=0.45, ymax=0.55, color='gray', linestyle='--')
    ax.axhline(y=6, xmin=0.78, xmax=0.88, color='gray', linestyle='--')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('1-Dimensional Plot')

    # Show the plot
    plt.show()