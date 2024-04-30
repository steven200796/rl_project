import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

iterations_key = 'timesteps'
scores_key = 'results'
log_threshold = 1e6
y_min = -21
y_max = 21
tstep_env_mult = 50
DPI = 400

fig_len = 8
fig_width = 6

def group_npz_files(dir):
    values = defaultdict(list)
    for basename in os.listdir(dir):
        if basename.endswith('.npz'):
            fp = os.path.join(dir, basename)
            data = np.load(fp)
            timesteps = data[iterations_key]
            scores = data[scores_key]
            for i, ts in enumerate(timesteps):
                values[ts].extend(scores[i])
    npz_obj = {iterations_key: values.keys(), scores_key: np.array(list(values.values()))
}
    return npz_obj

def plot(filepaths, xlabel, ylabel, title, symlog, no_shade, savepath):

    plt.figure(figsize=(fig_len, fig_width))
    for fp in filepaths: 
        if os.path.isdir(fp):
            data = group_npz_files(fp)
        else:
            data = np.load(fp)

        iterations = data[iterations_key]
        scores = data[scores_key]

        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)

        if not no_shade:
            # 95% confidence interval 2 std dev shade
            plt.fill_between(iterations, mean - 2 * std, mean + 2 * std, alpha=0.3)

        legend_label = os.path.splitext(os.path.basename(fp))[0]
        plt.plot(iterations, mean, label=f'{legend_label}')

    if symlog:
        plt.xscale('symlog', linthresh=log_threshold)

    plt.axhline(y=y_max, color='g', linestyle='--', linewidth=0.5) 
    plt.axhline(y=y_min, color='r', linestyle='--', linewidth=0.5)

    y_ticks = list(plt.yticks()[0]) + [y_min, y_max]

    plt.yticks(y_ticks)
    plt.ylim(y_min - 1, y_max + 1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title) 
    plt.legend()

    if savepath:
        plt.savefig(savepath, dpi=DPI)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot standard deviations of episode scores')
    parser.add_argument('file_paths', nargs='+', type=str, help='Paths to the .npz files')
    parser.add_argument('--xlabel', type=str, default='Iteration', help='Label for the x-axis')
    parser.add_argument('--ylabel', type=str, default='Mean score (over 30 episodes)', help='Label for the y-axis')
    parser.add_argument('--title', type=str, default='Scores', help='Title of the plot')
    parser.add_argument('--log_scale', action='store_true', help='Plot with a log scale cutoff')
    parser.add_argument('--no_shade', action='store_true', help='Don\'t shade 95% confidence interal')
    parser.add_argument('--save', type=str, default=None, help='Save path')
    args = parser.parse_args()



    plot(args.file_paths, args.xlabel, args.ylabel, args.title, args.log_scale, args.no_shade, args.save)

