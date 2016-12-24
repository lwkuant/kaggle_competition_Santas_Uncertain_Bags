# -*- coding: utf-8 -*-
"""
kaggle competition Santa's Uncertain Bags
"""

"""
Steps:
(1) 
"""


### EDA

# load required packages
import numpy as np 
import scipy as sp
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns

import os 
os.chdir(r'D:\Project\kaggle_competition_Santas_Uncertain_Bags')

# load the dataset 
gifts = pd.read_csv(r'gifts.csv/gifts.csv')

# check the dataset 
print(gifts.shape)
print(gifts.head())
print(gifts.info())
# the above codes show that the dataset is loaded correctly 

# get the count of each gift
gifts_name = list(gifts.GiftId)
gifts_type_name = [x.split('_')[0] for x in gifts_name]
gifts_type_name_unique = np.unique(gifts_type_name)
init_count = [0 for _ in range(len(gifts_type_name_unique))]
gifts_count = dict(((zip(gifts_type_name_unique, init_count))))
for gift in gifts_type_name:
    gifts_count[gift]+=1
print(gifts_count)

# simulate gifts weights
gifts_weights_distribution = {
    'horse': lambda: max(0, np.random.normal(5, 2, 1)[0]),
    'ball': lambda: max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
    'bike': lambda: max(0, np.random.normal(20, 10, 1)[0]),
    'train': lambda: max(0, np.random.normal(10, 5, 1)[0]),
    'coal': lambda: 47 * np.random.beta(0.5, 0.5, 1)[0],
    'book': lambda: np.random.chisquare(2, 1)[0],
    'doll': lambda: np.random.gamma(5, 1, 1)[0],
    'blocks': lambda: np.random.triangular(5, 10, 20, 1)[0],
    'gloves': lambda: (3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3
                       else np.random.rand(1)[0])}
                       
def simulate_weights(df, gifts_weights_distribution, seed):
    
    import numpy as np
    import collections
    import pandas as pd
    
    np.random.seed(seed)
    
    weights_dict_temp = collections.defaultdict(list)
    
    gifts_name = list(df.GiftId)
    gifts_type_name_name = [x.split('_')[0] for x in gifts_name]
    gifts_type_name_name_unique = np.unique(gifts_type_name_name)
    
    for gift in gifts_type_name_name:
        weights_dict_temp[gift].append(gifts_weights_distribution[gift]())
    
    weights_dict = dict()
    for type_name in gifts_type_name_name_unique:
        weights_dict[type_name] = pd.Series(weights_dict_temp[type_name], index=[type_name+'_'+str(i)
        for i in range(len(weights_dict_temp[type_name]))])
    
    return weights_dict

## plot the data 

# plot the count 
seed = 100
gift_weights = simulate_weights(gifts, gifts_weights_distribution, seed)

def visualize_count(df, gifts_weights_distribution, seed):
    import matplotlib.pyplot as plt 
    import matplotlib
    %matplotlib inline
    from matplotlib import cm
    from matplotlib import colors
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    gift_weights = simulate_weights(df, gifts_weights_distribution, seed)
    
    gift_type_names = list(gift_weights.keys())
    gift_counts = [len(gift_weights[type_name]) for type_name in gift_type_names]
    order_ind = np.array(gift_counts).argsort()[::-1]
    gift_counts = sorted(gift_counts, reverse=True)
    gift_type_names = [gift_type_names[i] for i in order_ind]
                   
    matplotlib.style.use('ggplot')    
    colormap = cm.get_cmap('Greens')     
    norm = colors.Normalize(vmax=max(gift_counts), vmin=min(gift_counts))      
    bar_colors = [colormap(norm(val)) for val in gift_counts]
    color_map = cm.get_cmap('Spectral')
    fig, axes = plt.subplots(figsize=[15, 15])
    axes.bar(list(range(len(gift_type_names))), gift_counts, color=bar_colors)
    axes.tick_params(labelsize=20)
    axes.set_xticks(np.arange(len(gift_type_names))+0.4)
    axes.set_xticklabels(gift_type_names, fontsize=20)
    axes.set_xlabel('Gift type_name', fontsize=25)
    axes.set_ylabel('Count', fontsize=25)
    axes.set_ylim(0, max(gift_counts)+100)
    axes.set_title('Count for Each Gift Type', fontsize=30)
    fig.tight_layout()
    plt.savefig('Count_for_each_type_name.png', dpi=300)

visualize_count(gifts, gifts_weights_distribution, seed)

# plot the each gift type_name's weight boxplot
seed = 100

def visualize_distribution_box(df, gifts_weights_distribution, seed):
    import matplotlib.pyplot as plt 
    import matplotlib
    %matplotlib inline
    from matplotlib import cm
    from matplotlib import colors
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    gift_weights = simulate_weights(df, gifts_weights_distribution, seed)
    
    gift_type_names = list(gift_weights.keys())
    gift_counts = [len(gift_weights[type_name]) for type_name in gift_type_names]

    fig, axes = plt.subplots(figsize=[15, 15])
    axes.tick_params(labelsize=20)
    sns.boxplot(pd.DataFrame(gift_weights))
    axes.set_title('Boxplot for Each Gift Type', fontsize=30)
    axes.set_xlabel('Gift Type', fontsize=25)
    axes.set_ylabel('Weights', fontsize=25)
    fig.tight_layout()
    plt.savefig('Boxplot_for_each_type_name.png', dpi=300)

visualize_distribution_box(gifts, gifts_weights_distribution, seed)
    
# plot the each gift type_name's weight distribution (KDE)
seed = 100

def visualize_distribution_kde(df, gifts_weights_distribution, seed):
    import matplotlib.pyplot as plt 
    import matplotlib
    %matplotlib inline
    from matplotlib import cm
    from matplotlib import colors
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    gift_weights = simulate_weights(df, gifts_weights_distribution, seed)
    
    gift_type_names = list(gift_weights.keys())
    gift_counts = [len(gift_weights[type_name]) for type_name in gift_type_names]

    color_palette = sns.color_palette(palette='Paired', n_colors=len(gift_type_names))
    fig, axes = plt.subplots(figsize=[15, 15])
    axes.tick_params(labelsize=20)
    for (type_name, col) in zip(gift_type_names, color_palette):
        sns.kdeplot(gift_weights[type_name], color=col, alpha=0.8, linewidth=3,
                    ax=axes, label=type_name)
    axes.legend(gift_type_names, loc=0, fontsize=15)
    axes.set_title('Kde for Each Gift Type', fontsize=30)
    axes.set_xlabel('Gift Type', fontsize=25)
    axes.set_ylabel('Density', fontsize=25)
    fig.tight_layout()
    plt.savefig('Kde_for_each_type_name.png', dpi=300)

visualize_distribution_kde(gifts, gifts_weights_distribution, seed)
    
# plot the each gift type_name's weight distribution (Histogram)
seed = 100

def visualize_distribution_histo(df, gifts_weights_distribution, seed):
    import matplotlib.pyplot as plt 
    import matplotlib
    %matplotlib inline
    from matplotlib import cm
    from matplotlib import colors
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    gift_weights = simulate_weights(df, gifts_weights_distribution, seed)
    
    gift_type_names = list(gift_weights.keys())
    gift_counts = [len(gift_weights[type_name]) for type_name in gift_type_names]
    
    fig = plt.figure(figsize=(20, 20))                
    plt.suptitle('Histogram for Each Gift Type', fontsize=25, y=0.94)
    for (i, type_name) in enumerate(gift_type_names, start=1):
        axes = fig.add_subplot(3, 3, i)
        axes.tick_params(labelsize=10)
        sns.distplot(gift_weights[type_name], ax=axes, color='#2E8B57')
        axes.set_xlabel(type_name, fontsize=15)
        axes.set_ylabel('Density', fontsize=15)
    #fig.tight_layout()
    plt.savefig('Histogram_for_each_type_name.png', dpi=300)    
 
visualize_distribution_histo(gifts, gifts_weights_distribution, seed)
       
# put the gifts in the boxes
seed=100
np.random.seed(100)
gifts_name = list(gifts.GiftId) # later I will shuffle the names as indicators
gift_weights = simulate_weights(gifts, gifts_weights_distribution, seed)

from collections import defaultdict
boxes_name = defaultdict(list)
for i in range(1, 1001):
    boxes_name[str(i)] = []

boxes_weight = dict()
for i in range(1, 1001):
    boxes_weight[str(i)] = 0

space = 20

from copy import deepcopy

gifts_name_shuffle = deepcopy(gifts_name)
gifts_name_for_remove = deepcopy(gifts_name)
np.random.shuffle(gifts_name_shuffle)
print(gifts_name_shuffle[:100])

import timeit
start_time = timeit.default_timer()
for gift in gifts_name_shuffle:
    weight = gift_weights[gift.split('_')[0]][int(gift.split('_')[1])]
    if weight >= 40:
        continue
    else:
        for num in range(1, 1001):
            if boxes_weight[str(num)]+weight <= 50-space:
                boxes_weight[str(num)] += weight
                boxes_name[str(num)].append(gift)
                gifts_name_for_remove.remove(gift)
                break
            else:
                continue
elapsed = timeit.default_timer() - start_time  
print(elapsed)
print('done')

# check the len must be equal or larger than 3
len_check = [len(boxes_name[num]) for num in sorted(list(boxes_name.keys()), key=lambda x: int(x))]
len_check = np.array(len_check)
print(np.argwhere(len_check <= 2))

print(gifts_name_for_remove)

reput_ind = np.argwhere(len_check <= 2)
for (ind, num) in enumerate(reput_ind):
    del boxes_weight[str(num[0]+1)]
    del boxes_name[str(num[0]+1)]
    
# submit the outcome
import os 
os.chdir(r'D:\Project\kaggle_competition_Santas_Uncertain_Bags')

file = open('sample_submission.csv', 'w')
file.write('Gifts\n')
for line in list(boxes_name.values()):
    file.write(' '.join(line)+'\n')
file.close()
