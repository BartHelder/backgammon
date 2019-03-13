import matplotlib as mpl
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file
import json
import os
import numpy as np
from collections import defaultdict

# Problem: not enough colors in standard mpl
def getColor(c, N, idx):
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N-1)
    return cmap(norm(idx))

def plot_lambdas(colormap):
    #  Add results to one dict
    results_l = []
    filelist = os.listdir('Results/run 2')
    for i in filelist:
        if i.endswith(".json") and i.startswith("results_t"):
            with open('Results/run 2/'+i, 'r') as f:
                tmp = json.load(f)
                tmp2 = []
                for j in tmp:
                    tmp2.append((int(j), tmp[j]))
                results_l.append((float(i[10:-5])/1000, tmp2))
    results_l.sort()

    fig1 = plt.figure(figsize=(8,6))
    index = 0
    for lamda, results in results_l:
        plt.plot(*zip(*results), label=lamda, color=getColor(colormap, len(results_l), index))
        index += 1
    plt.xticks()
    plt.yticks()
    plt.xlim((-500, 11800))
    plt.legend(title='$\lambda$ [-]')
    plt.xlabel("Episode [-]")
    plt.ylabel("Fraction of games won [-]")
    plt.show()

def plot_nhidden(colormap):
    # results on n_hidden
    results_h = []
    filelist = os.listdir('Results/run hidden')
    for i in filelist:
        if i.endswith(".json") and i.startswith("results_h"):
            with open('Results/run hidden/'+i, 'r') as f:
                tmp = json.load(f)
                tmp2 = []
                for j in tmp:
                    tmp2.append((int(j), tmp[j]))
                results_h.append((int(i[9:-5]), tmp2))
    results_h.sort()

    fig2 = plt.figure(figsize=(8,6))
    index = 0
    for n_hidden, results in results_h:
        plt.plot(*zip(*results), label=n_hidden, color=getColor(colormap, len(results_h), index))
        index += 1
    plt.xticks()
    plt.xlim(-500,10500)
    plt.legend(title='Hidden layer size [-]')
    plt.xlabel("Episode [-]")
    plt.ylabel("Fraction of games won [-]")
    plt.show()

def plot_alphalambda(colormap):
    ## merge alpha-lambda results:
    pre_results = defaultdict(dict)
    results_g = []
    filelist = os.listdir('Results/run al')
    for i in filelist:
        if i.endswith(".json"):
            with open('Results/run al/'+i, 'r') as f:
                tmp = json.load(f)['4000']
                parts = i[:-5].split('-')
                a,l = float(parts[0]), float(parts[1])
                pre_results[l][a] = tmp
    for i in pre_results:
        tmp2 = []
        for j in pre_results[i]:
            tmp2.append((j,pre_results[i][j]))
        tmp2.sort()
        results_g.append((i, tmp2))
    results_g.sort()

    fig3 = plt.figure(figsize=(8,6))
    index = 0
    for lamda, results in results_g:
        plt.plot(*zip(*results), label=lamda, color=getColor(colormap, len(results_g), index))
        index += 1
    plt.xscale('log')
    plt.legend(title='$\lambda$ [-]')
    plt.xlabel("Learning rate [-]")
    plt.ylabel("Fraction of games won [-]")
    plt.show()

def plot_compares(colormap):
    #  Add results to one dict
    results_m = []
    results_o = []
    filelist = os.listdir('Results/run compare')
    for i in filelist:
        with open('Results/run compare/'+i, 'r') as f:
            tmp = json.load(f)
            tmp2 = []
            for j in tmp:
                tmp2.append((int(j), tmp[j]))
            if i.startswith('results_t'):
                results_m.append((float(i[10:-5])/1000, tmp2))
            elif i.startswith('results_o'):
                results_o.append((float(i[10:-5])/1000, tmp2))

    results_m.sort()
    results_o.sort()

    fig4 = plt.figure(figsize=(8,6))

    index = 1
    for lamda, results in results_o:
        plt.plot(*zip(*results), label='$\lambda$='+str(lamda), linestyle='--', color=getColor(colormap, 20, index))
        index += 2
    index = 0
    for lamda, results in results_m:
        plt.plot(*zip(*results), label=lamda, color=getColor(colormap, 20, index))
        index += 2
    plt.xticks()
    plt.yticks()
    plt.xlim((-500, 10500))
    plt.legend(title='Feature extraction method \n Original           Modified', ncol=2)
    plt.xlabel("Episode [-]")
    plt.ylabel("Fraction of games won [-]")
    plt.show()

if __name__=="__main__":

    c = "coolwarm"
    # plot_lambdas(colormap=c)
    # plot_nhidden(colormap=c)
    # plot_alphalambda(colormap=c)
    plot_compares(colormap = 'tab20')

