import numpy as np
import daproli as dp

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()
sns.set_color_codes()


def plot_profile(ts_name, profile, true_cps=None, found_cps=None, show=True, score="roc_auc_score", save_path=None, font_size=26):
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(20, 5))

    ax.plot(np.arange(profile.shape[0]), profile, color='b')

    ax.set_title(ts_name, fontsize=font_size)
    ax.set_xlabel('split point  $s$', fontsize=font_size)
    ax.set_ylabel(score, fontsize=font_size)

    if true_cps is not None:
        dp.map(lambda true_cp: ax.axvline(x=true_cp, linewidth=2, color='r', label='True Change Point'), true_cps)

    if found_cps is not None:
        dp.map(lambda found_cp: ax.axvline(x=found_cp, linewidth=2, color='g', label='Found Change Point'), found_cps)

    if true_cps is not None or found_cps is not None:
        plt.legend(prop={'size': font_size})

    if show is True:
        ax.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_ts(ts_name, ts, true_cps=None, show=True, save_path=None, font_size=26):
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(20, 5))

    if true_cps is not None:
        segments = [0] + true_cps.tolist() + [ts.shape[0]]
        for idx in np.arange(0, len(segments)-1):
            ax.plot(np.arange(segments[idx], segments[idx+1]), ts[segments[idx]:segments[idx+1]])
    else:
        ax.plot(np.arange(ts.shape[0]), ts)

    ax.set_title(ts_name, fontsize=font_size)

    # if true_cps is not None:
        # ax.legend(prop={'size': font_size})

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_profile_with_ts(ts_name, ts, profile, true_cps=None, found_cps=None, show=True, score="Score", save_path=None, font_size=26):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': .05},  figsize=(20,10))

    if true_cps is not None:
        segments = [0] + true_cps.tolist() + [ts.shape[0]]
        for idx in np.arange(0, len(segments)-1):
            ax1.plot(np.arange(segments[idx], segments[idx+1]), ts[segments[idx]:segments[idx+1]])

        ax2.plot(np.arange(profile.shape[0]), profile, color='b')
    else:
        ax1.plot(np.arange(ts.shape[0]), ts)
        ax2.plot(np.arange(profile.shape[0]), profile)

    ax1.set_title(ts_name, fontsize=font_size)
    ax2.set_xlabel('split point  $s$', fontsize=font_size)
    ax2.set_ylabel(score, fontsize=font_size)

    for ax in (ax1, ax2):
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    if true_cps is not None:
        for idx, true_cp in enumerate(true_cps):
            ax1.axvline(x=true_cp, linewidth=2, color='r', label=f'True Change Point' if idx == 0 else None)
            ax2.axvline(x=true_cp, linewidth=2, color='r', label='True Change Point' if idx == 0 else None)

    if found_cps is not None:
        for idx, found_cp in enumerate(found_cps):
            ax1.axvline(x=found_cp, linewidth=2, color='g', label='Predicted Change Point' if idx == 0 else None)
            ax2.axvline(x=found_cp, linewidth=2, color='g', label='Predicted Change Point' if idx == 0 else None)

    ax1.legend(prop={'size': font_size})

    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")