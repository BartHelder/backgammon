import matplotlib as mpl
from matplotlib import pyplot as plt
import json
import os



#  Add results to one dict
results = []
filelist = os.listdir('Results')
for i in filelist:
    if i.endswith(".json") and i.startswith("results_t0"):
        with open('Results/'+i, 'r') as f:
            tmp = json.load(f)
            results.append((float(i[10:-5])/1000, tmp))

results.sort()

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

colormap = 'viridis'
plot_trace_stats(results, colormap)
