# ======================================================================================================================
#
#                  ___                                    ___          ___          ___                   ___
#       ___       /  /\                                  /  /\        /  /\        /  /\         ___     /  /\
#      /  /\     /  /::\                                /  /:/_      /  /::\      /  /::\       /  /\   /  /:/_
#     /  /:/    /  /:/\:\   ___     ___  ___     ___   /  /:/ /\    /  /:/\:\    /  /:/\:\     /  /:/  /  /:/ /\
#    /  /:/    /  /:/~/::\ /__/\   /  /\/__/\   /  /\ /  /:/ /::\  /  /:/  \:\  /  /:/~/:/    /  /:/  /  /:/ /::\
#   /  /::\   /__/:/ /:/\:\\  \:\ /  /:/\  \:\ /  /://__/:/ /:/\:\/__/:/ \__\:\/__/:/ /:/___ /  /::\ /__/:/ /:/\:\
#  /__/:/\:\  \  \:\/:/__\/ \  \:\  /:/  \  \:\  /:/ \  \:\/:/~/:/\  \:\ /  /:/\  \:\/::::://__/:/\:\\  \:\/:/~/:/
#  \__\/  \:\  \  \::/       \  \:\/:/    \  \:\/:/   \  \::/ /:/  \  \:\  /:/  \  \::/~~~~ \__\/  \:\\  \::/ /:/
#       \  \:\  \  \:\        \  \::/      \  \::/     \__\/ /:/    \  \:\/:/    \  \:\          \  \:\\__\/ /:/
#        \__\/   \  \:\        \__\/        \__\/        /__/:/      \  \::/      \  \:\          \__\/  /__/:/
#                 \__\/                                  \__\/        \__\/        \__\/                 \__\/
#
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
# ======================================================================================================================

""" --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------"""

''' External '''
import time
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import gzip
import pickle
import csv

'''  Internal '''
from TALLSorts.common import message, root_dir, create_dir
from TALLSorts.user import UserInput

''' --------------------------------------------------------------------------------------------------------------------
Global Variables
---------------------------------------------------------------------------------------------------------------------'''

label_colours = {
	'BCL11B':'#222222',
	'HOXA_KMT2A':'#F9DA49',
	'HOXA_MLLT10':'#91D44B',
	'NKX2':'#8E3CCE', 
	'TAL':'#DF3524',
	'TLX1':'#367BD8',
	'TLX3':'#57BFE0',
	'Diverse':'#ED75B2',
	'Unclassified':'#808080'
}

label_list = ['BCL11B', 'HOXA_KMT2A', 'HOXA_MLLT10', 'NKX2', 'TAL', 'TLX1', 'TLX3', 'Diverse']
label_list_unclassified = label_list + ['Unclassified']

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def run(ui=False):
    """
    A function that outputs a set of predictions and visualisations as per an input set of samples.
    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute TALLSorts, see UserInput class for further information.
    """

    if not ui:
        ui = UserInput()
        message(tallsorts_asci)

    # create output directory
    create_dir(ui.destination)
    # load classifier
    tallsorts = load_classifier()
    # run predictions
    run_predictions(ui, tallsorts)


def load_classifier(path=False):
    """
    Load the TALLSorts classifier from a pickled file.
    ...

    Parameters
    __________
    path : str
        Path to a pickle object that holds the TALLSorts model.
        Default: "/models/tallsorts/tallsorts.pkl.gz"

    Returns
    __________
    tallsorts : TALLSorts object
        TALLSorts object, unpacked, ready to go.
    """

    if not path:
        path = str(root_dir()) + "/models/tallsorts/tallsorts.pkl.gz"

    message("Loading classifier...")
    tallsorts = joblib.load(path)
    return tallsorts


def run_predictions(ui, tallsorts):
    """
    Use TALLSorts to make predictions
    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute TALLSorts, see UserInput class for further information.
    tallsorts : TALLSorts pipeline object

    Output
    __________
    Probabilities.csv
    Predictions.csv
    Multi_calls.csv
    Distributions.png and html
    Waterfalls.png and html
    (at the ui.destination path)

    """

    # fitting the data (which runs the classifier)
    tallsorts = tallsorts.fit(ui.samples)
    results = tallsorts.predict(ui.samples)

    # writing the probabilities to a CSV
    results.probs_raw_df.round(3).to_csv(f'{ui.destination}/probabilities.csv', index_label='Sample')

    # writing the highest predictions to a CSV
    pred_csv = results.calls_df[["y_pred"]].copy()
    pred_csv.columns = ["Predictions"]
    pred_csv.to_csv(f'{ui.destination}/predictions.csv', index_label='Sample')

    # writing multi calls to a CSV
    sample_order = sorted(results.multi_calls.keys(), key=lambda x: ui.samples.index.to_list().index(x))
    gen_multicall_csv(results.multi_calls, sample_order, f'{ui.destination}/multi_calls.csv')

    if ui.counts:
        message("Saving normalised/standardised counts.")
        processed_counts = results.transform(ui.samples)
        processed_counts["counts"].to_csv(ui.destination + "/processed_counts.csv")

    get_figures(results, ui.destination)

    message("Finished. Thanks for using TALLSorts!")

def gen_multicall_csv(multi_calls, sample_order, path):
    max_multicall = max([len(multi_calls[i]) for i in multi_calls])
    with open(path, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['']+[i for j in [[f'call_{k+1}', 'proba'] for k in range(max_multicall)] for i in j])
        for sample in sample_order:
            to_write = [sample]
            for call in multi_calls[sample]:
                to_write += [call[0], round(call[1],3)]
            to_write += ['' for i in range(2*max_multicall-len(to_write))]
            csvwriter.writerow(to_write)

def get_figures(results, destination, plots=["prob_scatter", "waterfalls"]):

    """
    Make figures of the results.
    ...

    Parameters
    __________
    samples : Pandas DataFrame
        Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
    destination : str
        Location of where the results should be saved.
    probabilities : Pandas DataFrame
        The result of running the get_predictions(samples, labels=False, parents=False) function.
        See function for further usage.
    plots : List
        List of plots required. Default:  "distributions", "waterfalls", and "manifold".
        See https://github.com/Oshlack/AllSorts/ for examples.

    Output
    __________
    Distributions.png, Waterfalls.png, Manifold.png at the ui.destination path.

    """

    message("Saving figures...")

    for plot in plots:

        if plot == "prob_scatter":
            dist_plot = gen_sample_wise_prob_plot(results.probs_raw_df, results.calls_df, labelThreshDict=None, batch_name=None, figsize=(800,600), return_plot=True)
            dist_plot.write_image(destination + "/prob_scatters.png", scale=2, engine="kaleido")
            dist_plot.write_html(destination + "/prob_scatters.html")

        if plot == "waterfalls":
            waterfall_plot = gen_waterfall_distribution(results.calls_df, labelThreshDict=None, batch_name=None, return_plot=True)
            waterfall_plot.write_image(destination + "/waterfalls.png", scale=2, engine="kaleido")
            waterfall_plot.write_html(destination + "/waterfalls.html")

        
def gen_sample_wise_prob_plot(probs_raw_df, calls_df, labelThreshDict=None, batch_name=None, figsize=(800,600), return_plot=False):
    """
    Given a set of predicted probabilities, generate a figure displaying distributions of probabilities. Analogous to `predict_dist` in ALLSorts.

    Essentially a visual representation of the probabilities.csv table.

    See https://github.com/Oshlack/TAllSorts/ for examples.
    ...

    Parameters
    __________
    probs_raw_df : Pandas DataFrame
        Table with samples (rows) and labels (columns). Entries are probabilities.
    calls_df : Pandas DataFrame
        A DataFrame with information about the top calls. Columns are: y_highest (highest call); proba_raw; proba_adj; y_pred (predicted call); multi_call (bool)
        Note that y_pred can be 'Unclassified', but y_highest will always be one of the labels.
    labelThreshDict : dict
        Dict of thresholds with labels as keys and threshold as values. Currently not used.
    batch_name : str
        String name of the batch to include in the title of the plot
    figsize : tuple
        Tuple of width and height of final image
    return_plot : bool
        Rather than showing the plot through whatever IDE is being used, send it back to the function call.
        Likely so it can be saved.

    Returns
    __________
    Plotly object containing the drawn figure

    Output
    __________
    Probability distribution figure.

    """

    if labelThreshDict is None:
        labelThreshDict = {i:0.5 for i in label_list}

    jitter = 0.5
    n_samples = len(probs_raw_df.index)

    fig = go.Figure()

    for i in range(n_samples):
        sample = probs_raw_df.index[i]
        sample_row = probs_raw_df.loc[sample]
        x = list(range(len(sample_row.index)))
        y = sample_row.to_list()
        colour = [label_colours[i] for i in sample_row.index]
        customdata = [[sample, i] for i in sample_row.index]
        fig.add_trace(go.Bar(x=x, y=y, width=0.9, 
                            marker={'color':'#90ee90'}, showlegend=False, visible=False,
                            customdata=customdata, hovertemplate='ID: %{customdata[0]}<br>%{customdata[1]}: %{y}<extra></extra>'))
    
    # add other points
    x = []
    y = []
    colour = []
    customdata = []
    for sample_no in range(n_samples):
        sample = probs_raw_df.index[sample_no]
        for label_no in range(len(label_list)):
            label_test = label_list[label_no]
            x.append(label_no + (np.random.random()-0.5)*jitter)
            y.append(probs_raw_df.loc[sample][label_test])
            colour.append('black')
            if calls_df.loc[sample]['y_pred'] in label_list:
                customdata.append([f'ID: {sample}<br>Call: {calls_df.loc[sample]["y_pred"]}<extra></extra>'])
            else:
                customdata.append([f'ID: {sample}<br>Call: {calls_df.loc[sample]["y_pred"]}<br>Highest: {calls_df.loc[sample]["y_highest"]}<extra></extra>'])

    fig.add_trace(go.Scatter(x=x, y=y,  mode="markers", marker={'color':colour, 'size':4}, visible=True, showlegend=False,
                            customdata=customdata, hovertemplate='%{customdata[0]}<extra></extra>'))

    # add threshold lines
    for x in range(len(label_list)):
            fig.add_shape(type="line",
                        x0=x-0.4, y0=labelThreshDict[label_list[x]],
                        x1=x+0.4, y1=labelThreshDict[label_list[x]],
                        line=dict(color="black", width=2), visible=True)

    # add dropdown
    fig.update_layout(
        updatemenus=[{
            'active':0,
            'buttons': [{'args':[{'visible':[False for j in range(n_samples)] + [True]}], 'label':'None', 'method':'update'}]
                    +[{'args':[{'visible':[j == i for j in range(n_samples)] + [True]}], 'label':probs_raw_df.index[i], 'method':'update'} for i in range(n_samples)],
            'direction':'down',
            'pad':{"r": 10, "t": 10},
            'showactive':True,
            'x':1, 'xanchor':"left",
            'y':0.8, 'yanchor':"top"
        }],
    )

    # adding dropdown text
    fig.update_layout(
        annotations=[{'text':"Select sample:", 'showarrow':False, 'x':1, 'xref':'paper', 'xanchor':"left", 'y':0.8, 'align':'left'}]
    )

    ticktext = label_list.copy()
    tickvals = [i for i in range(len(ticktext))]
    fig.update_xaxes(title_text='Classifier', showgrid=False, zeroline=False, 
                    tickmode = 'array', tickvals = tickvals, ticktext = ticktext, tickangle=45)
    fig.update_yaxes(title_text='Probability', range = (-0.01,1.01))
    fig.update_layout(title_text=f'Sample-wise classifier probabilities'+ (f': {batch_name} ({n_samples} samples)' if batch_name is not None else ''),
                    width=figsize[0],
                    height=figsize[1],
                    autosize=False,
                    template="plotly_white",
    )

    if return_plot:
        return fig
    else:
        fig.show()


def gen_waterfall_distribution(calls_df, labelThreshDict=None, batch_name=None, figsize=(1200,600), return_plot=False):

    """
    Given a set of predicted probabilities, generate a figure displaying the decreasing probabilities per sample. Analagous to `predict_waterfalls` and `_plot_waterfall` in ALLSorts

    This depiction is useful to compare probabilities more directly, in an ordered way, as to judge the efficacy
    of the classification attempt.

    See https://github.com/Oshlack/TAllSorts/ for examples.

    ...

    Parameters
    __________
    See descriptions in the gen_sample_wise_prob_plot function.

    Returns
    __________
    Plotly object containing the drawn figure

    Output
    __________
    Waterfalls figure.

    """

    if labelThreshDict is None:
        labelThreshDict = {i:0.5 for i in label_list}

    waterfallDf = calls_df.copy()
    waterfallDf.sort_values('proba_adj', inplace=True, ascending=False)
    waterfallDf.sort_values('y_pred', inplace=True, kind='stable', key=lambda x: pd.Series([label_list_unclassified.index(y) for y in x]))
    waterfallDf['colour'] = waterfallDf['y_pred'].apply(lambda x:label_colours[x])
    waterfallDf['sample_id'] = waterfallDf.index
    fig = go.Figure()

    x = 0
    for sample in waterfallDf.index:
        sample_row = waterfallDf.loc[sample]
        if sample_row["y_pred"] == 'Unclassified':
            hovertemplate = f'ID: {sample}<br>Call: {sample_row["y_pred"]}<br>Highest call: {sample_row["y_highest"]}<extra></extra>'
        else:
            hovertemplate = f'ID: {sample}<br>Call: {sample_row["y_pred"]}<extra></extra>'
        fig.add_trace(go.Bar(x=[x], y=[sample_row['proba_raw']], width=0.9, 
                            marker={'color':sample_row['colour']}, showlegend=False,
                            hovertemplate=hovertemplate))
        if sample_row['y_pred'] in label_list:
            fig.add_shape(type="line",
                        x0=x-0.4, y0=labelThreshDict[sample_row['y_pred']],
                        x1=x+0.4, y1=labelThreshDict[sample_row['y_pred']],
                        line=dict(color="black", width=2))
        x += 1

    # custom legend
    for label in sorted(label_colours.keys(), key=lambda x:label_list_unclassified.index(x)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker={'color':label_colours[label]}, showlegend=True, name=label))
    
    fig.update_xaxes(title_text='Samples', showgrid=False, showticklabels=False)
    fig.update_yaxes(title_text='Probability score', range = (0,1.01))
    fig.update_layout(title_text='Waterfall distribution' + (f': {batch_name} ({waterfallDf.shape[0]} samples)' if batch_name is not None else ''),
                    width=figsize[0],
                    height=figsize[1],
                    autosize=False,
                    template="plotly_white")
    fig.update_layout(legend=dict(title='Highest subtype call'))
    if return_plot:
        return fig
    else:
        fig.show()



''' --------------------------------------------------------------------------------------------------------------------
Global Variables
---------------------------------------------------------------------------------------------------------------------'''

tallsorts_asci = """                                                            
   .--------. ,---.  ,--.   ,--.    ,---.                  ,--.         
   '--.  .--'/  O  \ |  |   |  |   '   .-'  ,---. ,--.--.,-'  '-. ,---. 
      |  |  |  .-.  ||  |   |  |   `.  `-. | .-. ||  .--''-.  .-'(  .-' 
      |  |  |  | |  ||  '--.|  '--..-'    |' '-' '|  |     |  |  .-'  `)
      `--'  `--' `--'`-----'`-----'`-----'  `---' `--'     `--'  `----' 
    """
