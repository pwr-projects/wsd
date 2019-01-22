#%%
import os
import pandas as pd


#%%
show_scores = lambda filename: pd.DataFrame(pd.read_csv(f'scores_{filename}.csv').mean()).transpose()

#%%
show_scores('skladnica')
#%%
show_scores('skladnica_related')
#%%
show_scores('kpwr')
#%%
show_scores('kpwr_related')