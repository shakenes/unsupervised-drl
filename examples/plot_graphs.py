import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os
import re
from scipy import signal
from matplotlib import rc
plt.rc('font', family='serif',size='15')
rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None)
args = parser.parse_args()

matplotlib.rcParams.update({'font.size': 15})

assert args.dir is not None

config = [['Basic', 1000],
          ['DefendCenter', 4000],
          ['HealthGathering', 3500],
          ['HealthGatheringSupreme', 7500],
          ['MyWayHome', 6000]]

for env, max_len, legend_loc in zip(range(5), [6000, 4000, 10000, 5000, 2500], [4, 2, 2, 2, 2]):
    env_name = config[env][0]
    print(env_name)
    min_episodes = config[env][1]

    os.chdir(args.dir)

    keys = ['episode_reward']
    figsize = (6.0, 4.5)
    fig, axarr = plt.subplots(len(keys), sharex=False, figsize=figsize)
    sum = 0

    for filter, color, linetype in zip(['scratch', 'pretrained', 'convAE',      'SFA',   'combination_v2'],
                                       ['red',     'green',      'blue',   'magenta',      'black'],
                                       # [(2,0),     (1,1),        (5,1,1,1),(6,2,2,2,2,2), (5,4)]):
                                       ['', 'v', '^', 's', 'D']):
        data = []
        for file in os.listdir(args.dir):
            if re.match(r"^.*dqn_Vizdoom" + env_name + "-v0_log_"+filter+".json$", file):
                with open(file, 'r') as f:
                    tmp = json.load(f)
                    if len(tmp['episode']) > min_episodes:
                        data.append(tmp)
                        if 'episode' not in data[-1]:
                            raise ValueError('Log file "{}" does not contain the "episode" key.'.format(file))

        sum += len(data)
        print("Found {} datasets for {}.".format(len(data), filter))
        min_len = 10000
        for i in range(len(data)):
            min_len = min(min(len(data[i]['episode']), min_len), max_len)

        for idx, key in enumerate(keys):
            tmp = np.zeros(shape=(len(data), min_len))
            for i in range(len(data)):
                tmp[i][:] = data[i][key][0:min_len]
            std = signal.savgol_filter(np.std(tmp, axis=0), 251, 1)
            mean = signal.savgol_filter(np.mean(tmp, axis=0), 251, 1)
            axarr.plot(range(min_len)[0::20], mean[0::20], '-', marker=linetype, color=color, markevery=20)
            axarr.fill_between(range(min_len)[0::20], (mean+std)[0::20], (mean-std)[0::20], facecolors=color, alpha=0.1)
            axarr.set_ylabel(r'Reward')
            axarr.legend([r'Scratch', r'Pretrained', r'CAE', r'SFA', r'Pretr + CAE'], loc=legend_loc, fontsize=10)

    print("Found {} datasets total.".format(sum))
    plt.xlabel(r'Episodes')
    plt.tight_layout()
    plt.savefig('episodereward_{}.pdf'.format(env_name), bbox_inches='tight', format='pdf')
    plt.show()
