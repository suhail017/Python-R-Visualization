def myplot(df,themes,fig_size):
    '''This is a function to plot the multiple lines and boxplot of a dataframe. We are using the
    pandas to read csv file and seaborn to visualize
     
     Parameters:
    ===========
    df          = pandas.DataFrame,
    theme      =  "Whitegrid" or "Darkgrid" # Don't forget the String sign
    fig_size    = tuple (length, height); default: (16, 26),
                    controls the figure size of the output. '''
    
    # Importing the libraries
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Setting up the theme
    
    
    sns.set_style(themes)
    
    
    # lineplots
    
    plt.figure(figsize = fig_size)
    p = sns.lineplot(data = df, markers = True,)
    p.set_xlabel("Datapoints", fontsize = 20)
    p.set_ylabel("Number of Houses", fontsize = 20)
    plt.title('Line plots of the dataframe', fontdict={"size": 20})

    
    # Boxplots
    
    plt.figure(figsize = fig_size)
    q = sns.boxplot(data = df)
    q.set_xlabel("Models", fontsize = 20)
    q.set_ylabel("Number of Houses", fontsize=20)
    plt.title(f'Boxplot of the dataframe', fontdict={"size": 20})

    
    # Heatmap
    
    plt.figure(figsize = fig_size)
    sns.heatmap(df.corr(),
                cmap=sns.diverging_palette(3, 3, as_cmap=True),
                annot=True,
                fmt='.1f',
                linecolor='k',
                annot_kws={"size": 9},
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    plt.title(f'Features heatmap', fontdict={"size": 20})
