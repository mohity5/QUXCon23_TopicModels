from pandas import DataFrame as p_df
from pandas import concat as p_c

from sklearn.datasets import fetch_20newsgroups
X , y = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'), return_X_y= True)

def DataLoader():
    x = p_df(X)
    Y = p_df(y)
    data = p_c([Y,x], axis = 1)
    data.columns = ('TopicId','TextContent')
    
    #Shuffling the rows, if the data rows might be arranged
    data = data.sample(data.shape[0], ignore_index=True, random_state=2)

    return data