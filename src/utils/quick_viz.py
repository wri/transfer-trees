#!/usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
import rasterio as rs
import hickle as hkl
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
import pandas as pd
import re
# from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Base
# └── basic visualizations
#     ├── heatmaps
#     │   ├── multiply.py
#     │   ├── divide.py
#     ├── histograms
#     ├── confusion matrix

def style_axis(ax, 
               xlabel: str, 
               ylabel: str, 
               xgrid: bool,
               ygrid:bool,
               title: str = None, 
               grid_color: str = '#DDDDDD',
               tick_format: str = None,
               fontsize: int = None):
    """
    Applies consistent styling to the axes, including labels and gridlines.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to be styled.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    gridlines (bool): Option to add gridlines to the chart background.
    title (str, optional): The title of the chart.
    grid_color (str, optional): The color of the gridlines. Default is '#DDDDDD'.
    """
    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Style the bottom and left spines
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    
    # Set gridlines and axis below the plot elements
    ax.yaxis.grid(ygrid, color=grid_color)
    ax.xaxis.grid(xgrid,color=grid_color)
    ax.set_axisbelow(True)
    
    # Set the axis labels
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=15, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=15, fontsize=fontsize)
    if tick_format is not None:
        ax.ticklabel_format(style='plain', axis='y')

    # Optionally set the title
    if title:
        title_fontsize = fontsize + 2 if fontsize else 14
        ax.set_title(title, fontsize=title_fontsize, pad=15)

def heat_multiplot(matrices, cbarmin, cbarmax,  nrows = 13, ncols = 6):
    '''
    Type: Seaborn heatmap
    Purpose: Create a multiplot of heatmaps from a collection of matrices.

    Parameters:
    - matrices (array-like): A collection of matrices to be visualized as heatmaps.
    - cbarmin (float): The minimum value for the color bar scale.
    - cbarmax (float): The maximum value for the color bar scale.
    - nrows (int, optional): Number of rows for the subplot grid. Default is 13.
    - ncols (int, optional): Number of columns for the subplot grid. Default is 6.

    This function creates a multiplot of heatmaps from a collection of matrices. This function is most
    helpful if you want to visualize 3D arrays. It arranges the heatmaps in a grid with the specified 
    number of rows and columns. The color scale for the heatmaps is defined by the `cbarmin` 
    and `cbarmax` parameters. Each heatmap is displayed within its own subplot.

    Returns:
    None
    
    '''
    fig, axs = plt.subplots(ncols = ncols, nrows = nrows)
    fig.set_size_inches(18, 3.25*nrows)
    
    # create a list of indices from 0 to nrows*ncols
    to_iter = [[x for x in range(i, i + ncols + 1)] for i in range(0, nrows*ncols, ncols)]
    counter = 0
    
    #
    for r in range(1, nrows + 1):
        min_i = min(to_iter[r-1])
        max_i = max(to_iter[r-1])
        
        for i in range(ncols):
            sns.heatmap(data = matrices[..., counter], 
                        ax = axs[r - 1, i], 
                        cbar = True, 
                        vmin = cbarmin, # this could also be min_i
                        vmax = cbarmax, # this could also be max_i
                        cmap = sns.color_palette("viridis", as_cmap=True))
            axs[r - 1, i].set_xlabel("")
            axs[r - 1, i].set_ylabel("")
            axs[r - 1, i].set_yticks([])
            axs[r - 1, i].set_xticks([])
            counter += 1
        
    plt.show
    return None

def heat_compare_preds(location: str, tile_idx_a: tuple, tile_idx_b: tuple):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize multi-class predictions for two specific tiles in a given location 
    using heatmaps.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title to be displayed for the comparison plot.

    This function loads prediction data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b)
    from the specified location directory and creates side-by-side heatmaps to visualize the predictions. The
    heatmaps display values in the range [0, 1, 2], and each tile's heatmap is shown in a separate subplot.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    preds_a = rs.open(f'../tmp/{location}/preds/{str(x_a)}X{str(y_a)}Y_preds.tif').read(1)
    preds_b = rs.open(f'../tmp/{location}/preds/{str(x_b)}X{str(y_b)}Y_preds.tif').read(1)
    

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(preds_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2, 3]}, 
                vmin=0, vmax=3).set_title('Tile: ' + str(tile_idx_a))
    plt.subplot(1,2,2)
    sns.heatmap(preds_b, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2, 3]}, 
                vmin=0, vmax=3).set_title('Tile: ' + str(tile_idx_b));

    return None

def heat_compare_ard(location, tile_idx_a, tile_idx_b):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize two files (could be s2 data, ARD, feats, etc.)
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')

    plt.figure(figsize=(25,20))
    for plot in range(1, 14):
        for i in range(0, ard_a.shape[-1]):
            plt.subplot(3,5,plot)
            sns.heatmap(ard_a[..., i], 
                        xticklabels=False, 
                        yticklabels=False,
                        cbar_kws = {'ticks' : [ard_a.min(), ard_a.max()]}).set_title(f"index: {i}")
                
            # plt.subplot(3,5,2)
            # sns.heatmap(ard_b, 
            #             xticklabels=False, 
            #             yticklabels=False,
            #             cbar_kws = {'ticks' : [ard_b.min(), ard_b.max()]}).set_title(title_b);

    return None

def heat_combine_neighbors(arr_a, arr_b, index, title):
    '''
    Type: Seaborn heatmap
    Purpose: Combines neighboring tiles into a single array and plots a 
    heatmap in order to analyze artifacts and the effects of resegmentation.
    arr_a is left
    arr_b is right
    
    '''
    slice_a = arr_a[..., index]
    slice_b = arr_b[..., index]
    assert slice_a.shape[0] == slice_b.shape[0], "Slices must have the same height to be concatenated."

    comb_arr = np.concatenate((slice_a, slice_b), axis=1)

    plt.figure(figsize=(9,3))

    sns.heatmap(comb_arr, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [min(arr_a.min(), arr_b.min()), max(arr_a.max(), arr_b.max())]},
                vmin=min(slice_a.min(), slice_b.min()),
                vmax=max(slice_a.max(), slice_b.max()),
                ).set_title(str(title))

    #plt.close()
    return None


def cm_roc_pr(model, y_test, pred, probs_pos):

    ''' 
    Visualize the performance of a classification model using a Confusion Matrix, 
    ROC Curve, and Precision-Recall Curve.

    Parameters:
    - model: The trained classification model.
    - y_test: True labels of the test set.
    - pred: Predicted labels of the test set.
    - probs_pos: Probability of the positive class for each sample in the test set.

    Note:
    This function requires the scikit-learn library for confusion matrix visualization
    and matplotlib for creating ROC and Precision-Recall curves.
    '''
    
    with open(f'../models/{model}.pkl', 'rb') as file:  
        model = pickle.load(file)

     # Calculate and plot CM
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot();

    # Calculate and plot ROC AUC 
    fpr, tpr, thresholds = roc_curve(y_test, probs_pos)

    plt.figure(figsize=(17,6)) 

    plt.subplot(1,2,1)
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model, color='green')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');

    
    # Calculate and plot precision-recall curve and no skill
    fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)
    no_skill = len(y_test[y_test == 1]) / len(y_test)

    plt.subplot(1,2,2)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model, color='purple')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None



def roc_curve_comp(X_test, y_test, model_names):

    '''
    Plot ROC Curves for multiple classification models.

    Parameters:
    - X_test: Testing features.
    - y_test: True labels of the test set.
    - model_names: List of models to be plotted and compared.

    Note:
    This function requires scikit-learn for calculating ROC curves.
    '''
    
    plt.figure(figsize=(17,6)) 
    
    # ROC curve
    for m in model_names:
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,1)
        
        # calculate and plot ROC curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
    
    # PR curve
    for m in model_names:
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,2)

        # calculate and plot precision-recall curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.4, 1.05])
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None


def learning_curve_comp(model_names, X_train, y_train, x_max):

    '''
    Plot learning curves to compare the performance of machine learning models.

    Parameters:
    - model_names: List of model names to be compared.
    - X_train: Training features.
    - y_train: True labels of the training set.
    - x_max: Maximum number of training samples to display on the x-axis.

    Note: This function requires scikit-learn for learning curve computation
    '''
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 
    
    plt.figure(figsize = (13,6))
    for i, x in zip(model_names, colors[:len(model_names)+1]):

        filename = f'../models/{i}.pkl'

        with open(filename, 'rb') as file:
            model = pickle.load(file)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, 
                                                                              X_train, 
                                                                              y_train, 
                                                                              cv=5, 
                                                                              return_times=True,
                                                                              verbose=0) 

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.grid()
        plt.plot(train_sizes, train_scores_mean, "x-", color=x, label=f"{i[0:4]} Train")
        plt.plot(train_sizes, test_scores_mean, ".-", color=x, label=f"{i[0:4]} Test")
    
    plt.xlim([1000, x_max])
    plt.ylim([0.0, 1.2])
    plt.title(f'Learning Curve Comparison for {len(model_names)} Models')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(title='Models', loc='lower right');        
        
    return None


def learning_curve_catboost(X_train_all,
                            X_train_dropped,
                            y_train,
                            output_file: str = None):

    '''
    Plots the learning curve comparing 2 catboost models:
    - one uses sentinel imagery and TTC features 
    - one uses only sentinel imagery

    Parameters:
    - X_train_all: Training data including sentinel imagery and transferred features 
    - X_train_dropped: Training data including only sentinel
    - y_train: True labels of the training set.
        
    '''
    
    colors = ['royalblue', 'maroon']

    fig, ax = plt.subplots(figsize=(12, 7))   

    # Initialize CatBoost model
    catboost = CatBoostClassifier(verbose=0, iterations=300)

    # Plot learning curve for model with TTC features
    train_sizes_ttc, train_scores_ttc, test_scores_ttc = learning_curve(
        catboost, X_train_all, y_train, cv=5, return_times=False, verbose=0,
        # train_sizes = [10000, 40000, 70000, 100000] 
    )
    train_scores_mean_ttc = np.mean(train_scores_ttc, axis=1)
    test_scores_mean_ttc = np.mean(test_scores_ttc, axis=1)
    ax.plot(train_sizes_ttc, train_scores_mean_ttc, "o--", color=colors[0], label="Train (with TL)")
    ax.plot(train_sizes_ttc, test_scores_mean_ttc, "o-", color=colors[0], label="Test (with TL)")

    # Plot learning curve for model without TTC features
    train_sizes, train_scores, test_scores = learning_curve(
        catboost, X_train_dropped, y_train, cv=5, return_times=False, verbose=0
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    ax.plot(train_sizes, train_scores_mean, "o--", color=colors[1], label="Train (without TL)")
    ax.plot(train_sizes, test_scores_mean, "o-", color=colors[1], label="Test (without TL)")

    # Formatting the plot
    ax.axhline(y=0.80, color='red', linestyle='--', label='Target Accuracy')
    ax.grid()
    ax.xlim([train_sizes.min() - 10000, train_sizes.max() + 10000])
    ax.ylim([0.0, 1.2])
    ax.title("Learning Curve Comparison")
    ax.xlabel("Number of Training Samples")
    ax.ylabel("Score")
    ax.legend(title="CatBoost Model", loc="lower right")

    ax.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    ax.show()

    return None


def roc_auc_curve_catboost(X_train_all, X_train_dropped, X_test, y_train, y_test, output_file: str = None):
    '''
    Plots an ROC AUC curve comparing 2 CatBoost models:
    - one using sentinel imagery and TTC features 
    - one using only sentinel imagery
    
    Parameters:
    - X_train_all: Training data with sentinel imagery and transferred features 
    - X_train_dropped: Training data with only sentinel imagery
    - X_test: Test data used for evaluation
    - y_train: Training labels
    - y_test: Test labels
    - output_file: Optional path to save the plot
    '''

    plt.figure(figsize=(8,5)) 
    
    # Initialize CatBoost models
    models = {
        "With Transfer Learning": X_train_all,
        "Without Transfer Learning": X_train_dropped
    }

    for label, X_train in models.items():
        catboost = CatBoostClassifier(verbose=0, iterations=300)
        catboost.fit(X_train, y_train)
        
        # Get predicted probabilities for the positive class
        probs = catboost.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, marker=',', label=f'{label} (AUC = {roc_auc:.2f})')
    
    # Plot the no-skill line
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC AUC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()


def horizontal_stacked_bar(df: pd.DataFrame, 
                           region: str,
                           color_dict: dict,
                           sort_by: str,
                           dpi: int = 300,
                           alpha: float = 1.0,
                           figsize: tuple = (13, 8)):
    """
    Creates a horizontal 100% stacked bar chart showing the percentage 
    area for different tree cover classes per district in a given region.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    region (str): Region name to filter.
    color_dict (dict): Dictionary mapping classes to colors.
    sort_by (str): Column to sort by.
    output_file (str): Path to save the plot.
    dpi (int): Resolution of saved figure.
    alpha (float): Transparency of bars.
    figsize (tuple): Size of the figure.
    """
    df = df.iloc[0:-2]
    df = df[df["Zone"] == region]
    categories = ['Agroforestry (ha)', 'Natural (ha)', 'Monoculture (ha)', 'Background (ha)']
    clean_categories = [re.sub(r"\s*\(.*?\)", "", c) for c in categories]

    # Normalize to percentages
    df[categories] = df[categories].div(df[categories].sum(axis=1), axis=0) * 100

    # Sort by the chosen column
    df = df.sort_values(by=sort_by, ascending=False)
    fontsize=14

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    left = np.zeros(len(df))

    for cat_label, cat_col in zip(clean_categories, categories):
        bars = ax.barh(
            df["District"], 
            df[cat_col], 
            left=left, 
            label=cat_label,  # <-- clean label
            color=color_dict.get(cat_label, "#cccccc"),
            alpha=alpha
        )
        for bar, value in zip(bars, df[cat_col]):
            if value > 3:
                ax.text(bar.get_x() + .4,
                        bar.get_y() + bar.get_height() / 2,
                        f'{value:.0f}%',
                        va='center', ha='left',
                        color='black', fontsize=fontsize)
        left += df[cat_col].values

    # Style the chart
    style_axis(
        ax=ax,
        xlabel=" ",
        ylabel=" ",
        title=f" ",
        xgrid=True,
        ygrid=False,
        fontsize=fontsize,
    )
    ax.set_xlim(0, 100)
    ax.legend(title="System", loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=fontsize)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=fontsize)


    plt.tight_layout()
    df.to_csv(f'../../data/figures/h_stacked_bar_{region}_data.csv')
    plt.savefig(f'../../data/figures/h_stacked_bar_{region}test.png', dpi=dpi, bbox_inches='tight')
    plt.show()



def vertical_stacked_bar(df: pd.DataFrame, 
                title: str,
                color_dict: dict,
                categories: list,
                output_file: str = None,
                dpi: int = 300
               ):
    """
    Creates a stacked bar chart showing the total area in hectares for different tree cover classes per district.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    title (str): Title of the chart.
    color_dict (dict): Dictionary mapping classes to colors.
    categories (list): List of land use class columns to include in the stacked bar.

    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize the bottom for stacking
    bottom = np.zeros(len(df))

    # Plot each category
    for category in categories:
        ax.bar(
            df.district, 
            df[category], 
            bottom=bottom, 
            label=category, 
            color=color_dict.get(category, "#cccccc"),
        )
        bottom += df[category].values

    # Style the chart
    style_axis(
        ax=ax,
        xlabel="District",
        ylabel="Total Area (ha)",
        title=title,
        gridlines=True
    )

    # Rotate x-axis labels for readability
    ax.set_xticklabels(df.district, rotation=55, ha="right")
    ax.legend(title="System", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_feature_importance(df: pd.DataFrame, 
                            figsize: tuple = (8, 10),
                            fontsize= 12,
                            dpi: int = 300,
                            output_file = None):
    """
    Plots a horizontal bar chart of selected features by their importance scores using matplotlib.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'feature_names', 'feature_importance', 'category', and 'selected' columns.
    - figsize (tuple): Size of the figure (width, height).
    """

    # Filter to selected features only and sort
    df_selected = df[df["selected"] == True].copy()
    df_selected = df_selected.sort_values(by="feature_importance", ascending=True)

    # Update TTC label to 'Extracted Tree Features'
    df_selected["category"] = df_selected["category"].replace({"TTC": "TTC Embeddings"})

    # Define an updated color palette
    palette = {
        "Sentinel 2": "#4195d2",                    # cool blue
        "Sentinel 1": "#9e9ac8",                    # muted purple
        "TTC Embeddings": "#037d34ab",          # deep green
        "DEM": "#fe8266",                           # light orange
        "Texture (green band)": "#93d95d",          # more distinct green
        "Texture (red band)": "#d73027",            # dark red
        "Texture (NIR band)": "#fdb863"             # warm yellow-orange
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(df_selected))
    colors = [palette.get(cat, "#cccccc") for cat in df_selected['category']]

    bars = ax.barh(
        y=y_pos,
        width=df_selected['feature_importance'],
        color=colors,
    )

    # Axis settings
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_selected['feature_names'])
    ax.set_ylim(-0.5, len(df_selected) - 0.5)

    style_axis(
        ax=ax,
        xlabel="Feature Importance Score",
        ylabel="Feature Index",
        title=" ",
        xgrid=False,
        ygrid=True,
        fontsize=fontsize,
    )

    # Add legend manually
    unique_cats = df_selected["category"].unique()
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[cat]) for cat in unique_cats]
    ax.legend(handles, unique_cats, title="Feature Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()