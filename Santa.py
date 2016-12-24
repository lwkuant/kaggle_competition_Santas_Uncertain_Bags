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
    gifts_type_name = [x.split('_')[0] for x in gifts_name]
    gifts_type_name_unique = np.unique(gifts_type_name)
    
    for gift in gifts_type_name:
        weights_dict_temp[gift].append(gifts_weights_distribution[gift]())
    
    weights_dict = dict()
    for type in gifts_type_name_unique:
        weights_dict[type] = pd.Series(weights_dict_temp[type], index=[type+'_'+str(i)
        for i in range(len(weights_dict_temp[type]))])
    
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
    
    gift_types = list(gift_weights.keys())
    gift_counts = [len(gift_weights[type]) for type in gift_types]
    order_ind = np.array(gift_counts).argsort()[::-1]
    gift_counts = sorted(gift_counts, reverse=True)
    gift_types = [gift_types[i] for i in order_ind]
                   
    matplotlib.style.use('ggplot')    
    colormap = cm.get_cmap('Greens')     
    norm = colors.Normalize(vmax=max(gift_counts), vmin=min(gift_counts))      
    bar_colors = [colormap(norm(val)) for val in gift_counts]
    color_map = cm.get_cmap('Spectral')
    fig, axes = plt.subplots(figsize=[15, 15])
    axes.bar(list(range(len(gift_types))), gift_counts, color=bar_colors)
    axes.tick_params(labelsize=20)
    axes.set_xticks(np.arange(len(gift_types))+0.4)
    axes.set_xticklabels(gift_types, fontsize=20)
    axes.set_xlabel('Gift Type', fontsize=25)
    axes.set_ylabel('Count', fontsize=25)
    axes.set_ylim(0, max(gift_counts)+100)
    axes.set_title('Count for Each Gift Type', fontsize=30)
    fig.tight_layout()
    plt.savefig('Count_for_each_type.png', dpi=300)
    
# plot the each gift type's weight distribution




    