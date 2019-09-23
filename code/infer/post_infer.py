import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np


def view_burnin_plot(sampler, first_param, second_param):
    plt.style.use('seaborn-deep')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}

    fig, (ax0, ax1) = plt.subplots(2)
    ax0.set(ylabel='f(d' + str(first_param) + ')')
    ax1.set(ylabel='f(d' + str(second_param) + ')')

    for j in range(10):
        sns.tsplot(sampler.chain[j, :, first_param], ax=ax0)
        sns.tsplot(sampler.chain[j, :, second_param], ax=ax1)
    # plots 2 inferred parameters at a time


def chain(sampler, step_to_chain_at, ndim):
    return sampler.chain[:, step_to_chain_at:,:]


def create_dataframe(chained_sampler, param_num):
    traces = chained_sampler.reshape(-1, param_num).T
    samples_dictionary = {}
    for i in range(param_num-1):
        samples_dictionary["f"+str(i)] = traces[i]
    samples_df = pd.DataFrame(samples_dictionary)
    return samples_df


def get_infer_f(quantiled_samples, m):
    array = np.zeros(m)
    for i in range(m):
        a = quantiled_samples.get("f"+str(i)).values
        array[i] = a[0]
    return array


def get_beta(chained_sampler, ndim):
    traces = chained_sampler.reshape(-1, ndim).T
    beta = {'beta': traces[ndim-1]}
    beta_df = pd.DataFrame(beta)
    return beta_df
