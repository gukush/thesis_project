import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr



# based on: https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance

def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues.at[r, c] = pearsonr(tmp[r], tmp[c])[1] 
    pvalues = pvalues.astype(float)
    return pvalues


def calculate_tvalue(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = pearsonr(tmp[r], tmp[c])[1]
    return pvalues



our_dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(our_dir,'../metric_results_merged.csv'))

exclusion_list = ['Person','Fragment','Type']
exclusion_list_2 = ['Person','Fragment','Type',
                    'ROUGE-1','ROUGE-2','ROUGE-L',
                    'ROUGE-2-pre','ROUGE-WE-2','ROUGE-WE-3']
# Using list comprehension to select columns not in exclusion_list
metric_columns = [col for col in df.columns if col not in exclusion_list]
metric_columns_2 = [col for col in df.columns if col not in exclusion_list_2]
if False:
    for our_type in ('Extractive','Abstractive'):
        corr_df = df[df['Type']==our_type].copy()
        corr_df = corr_df[metric_columns_2].copy()
        
        for column in corr_df.columns:
            corr_df[column] = corr_df[column].str.replace(',', '.').astype(float)
        corr = corr_df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Use the "rocket" colormap
        cmap = sns.color_palette("Spectral", as_cmap=True)

        #sns.color_palette("rocket", as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio # vmax =.3 center=0.5, 
        sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=0.6, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

        fig = sns_plot.get_figure()
        fig.savefig(f"corr_heatmap_{our_type}.png")


if True:
    for our_type in ('Extractive','Abstractive'):
        corr_df = df[df['Type']==our_type].copy()
        corr_df = corr_df[metric_columns_2].copy()
        
        for column in corr_df.columns:
            corr_df[column] = corr_df[column].str.replace(',', '.').astype(float)
        corr = calculate_pvalues(corr_df)
        corr = np.log10(corr)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Use the "rocket" colormap
        cmap = sns.color_palette("Spectral", as_cmap=True)

        #sns.color_palette("rocket", as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio # vmax =.3 center=0.5, 
        sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=0.6, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

        fig = sns_plot.get_figure()
        normalized_significance = np.log10(0.001)

        insignificant = corr > normalized_significance
        filtered_df = corr[insignificant]
        print(np.where(insignificant))

        fig = sns_plot.get_figure()

        for ax in fig.axes:
            if ax.get_label() == '<colorbar>':
                ax = ax.axhline(normalized_significance,c='black',linewidth=2)



        fig.savefig(f"pvalue_heatmap_{our_type}.png")


if False:
    corr_df = df[metric_columns].copy()

    for column in corr_df.columns:
        corr_df[column] = corr_df[column].str.replace(',', '.').astype(float)

    corr = calculate_pvalues(corr_df)
    corr = np.log10(corr)
    #print(corr)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
      # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Use the "rocket" colormap
    cmap = sns.color_palette("Spectral", as_cmap=True)

    #sns.color_palette("rocket", as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio # vmax =.3 center=0.5, 
    sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=0.6, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

    #significance_level = np.log(0.05)
    normalized_significance = np.log10(0.001)
    #    #normalized_significance = norm(significance_level)
    #print(normalized_significance)
    #(tmp_min,tmp_max) =  sns_plot.collections[0].get_clim()
    #norm = plt.Normalize(vmin=tmp_min, vmax=tmp_max)
    #normalized_significance = norm(normalized_significance)
    insignificant = corr > normalized_significance
    filtered_df = corr[insignificant]
    print(np.where(insignificant))
    #sns_plot.collections[].xticks(list(plt.xticks()[0]) + extraticks)
    fig = sns_plot.get_figure()
    #cb = fig.axes
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
    #        #print(tmp)
    #        print(dir(ax))
    #        (tmp_min,tmp_max) = ax.collections[0].get_clim()
            ax = ax.axhline(normalized_significance,c='black',linewidth=2)
    
    #cb.ax.axhline(np.log(0.05), c='w')   #red_patch = plt.Line2D([0], [0], color='red', lw=2, label='Significance Level (ln(0.05))')
    #fig.legend(handles=[red_patch])
    #fig.canvas.draw()
    fig.savefig("pvalue_heatmap.png")

if False:
    corr_df = df[metric_columns].copy()


    for column in corr_df.columns:
        corr_df[column] = corr_df[column].str.replace(',', '.').astype(float)
    corr = corr_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Use the "rocket" colormap
    cmap = sns.color_palette("Spectral", as_cmap=True)

    #sns.color_palette("rocket", as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio # vmax =.3 center=0.5, 
    sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, #vmin=0.6, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

    fig = sns_plot.get_figure()
    fig.savefig("corr_heatmap.png")
