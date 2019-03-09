import matplotlib as mpl
from matplotlib import pyplot as plt
import json
import os
from collections import defaultdict


#  Add results to one dict
def merge_results(folder, id):
    results = []
    filelist = os.listdir('Results/'+folder)
    for i in filelist:
        if i.endswith(".json") and i.startswith("results_"+id):
            with open('Results/'+folder+'/'+i, 'r') as f:
                tmp = json.load(f)
                results.append((float(i[10:-5])/1000, tmp))

    results.sort()
    return results

## merge alpha-lambda results:
results_al = defaultdict(list)
filelist = os.listdir('Results/run al')
for i in filelist:
    if i.endswith(".json"):
        with open('Results/run al/'+i, 'r') as f:
            tmp = json.load(f)['4000']
            parts = i[:-5].split('-')
            a,l = float(parts[0]), float(parts[1])
            results_al[l].append((a, tmp))
for item in results_al:
    results_al[item].sort()

# Problem: not enough colors in standard mpl
def getColor(c, N, idx):
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N-1)
    return cmap(norm(idx))


def plot_trace_stats(input, c):
    """
    Compares the performance of different settings of the trace decay parameter lambda
    :param input: list of lambda, results tuples
    :param c: colormap used for plotting (str)
    :return:
    """

    fig1 = plt.figure(figsize=(10,6))
    plt.legend()
    index = 0
    for key, value in input:
        plt.plot(*zip(*value.items()), label=key, color=getColor(c, len(input), len(input)-index))
        index += 1
    plt.legend(title='$\lambda$ [-]')
    plt.xlabel("Episode (-)")
    plt.ylabel("Fraction of games won (-)")
    plt.show()


def plot_alpha_lambdas(input, c):
    fig1 = plt.figure(figsize=(10,6))
    plt.legend()
    index = 0
    for key, value in input:
        plt.plot(*zip(*value.items()), label=key, color=getColor(c, len(input), len(input)-index))
        index += 1
    plt.legend(title='$\lambda$ [-]')
    plt.xlabel("Learning rate $\alpha$ (-)")
    plt.ylabel("Fraction of games won (-)")
    plt.show()



if __name__ == "__main__""":

    colormap = 'viridis'
    plot_trace_stats(merge_results('run 2', 't'), colormap)
    plot_alpha_lambdas(results_al, colormap)


