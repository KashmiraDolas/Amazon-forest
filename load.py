import sys
import os
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
#Seaborn is a Python visualization library based on matplotlib.
# It provides a high-level interface for drawing attractive statistical graphics.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy import ndimage
from IPython.display import display

def data_process():
    df = pd.read_csv("input/train_v2.csv")


#count the unique labels
    from itertools import chain
    labels_list = list(chain.from_iterable([tags.split(" ") for tags in df['tags'].values]))
    labels_set = set(labels_list)
    #print(labels_set)
    print("There is {} unique labels including {}".format(len(labels_set), labels_set))


# Histogram of label instances

    labels_s = pd.Series(labels_list).value_counts()
    
    #print(labels_s)
    fig, ax = plt.subplots()
    p = sns.barplot(x=labels_s, y=labels_s.index)
    p.set_yticklabels(labels_s.index, rotation=45)
    plt.savefig('hist.png')
    plt.show()
    
    labels_l = []
    for i in df.tags.values:
        #print (i)
        for l in i.strip().split(" "):
            if l not in labels_l:
                labels_l.append(l)

#binary labels
    #print("converting it to binary labels")
    # print(list(labels_df))
    #print(labels_l)
    for l in labels_l:
        df [l] = df['tags'].apply(lambda x: 0 if l not in  x.split(' ') else 1)
    #print(df.head())
    df.to_csv("one_hot.csv", sep=',')
    return labels_l, df, labels_set

def make_cooccurence_matrix(labels_l, df):
    fig, ax = plt.subplots()
    numeric_df = df[labels_l];
    c_matrix = numeric_df.T.dot(numeric_df)
    p = sns.heatmap(c_matrix, annot=False, fmt="g", cmap='viridis')
    p.set_xticklabels(p.get_xticklabels(), rotation=35, fontsize=8)
    plt.show()
    return c_matrix




def main():
    labels_l, df, labels_set = data_process()
    c_matrix = make_cooccurence_matrix(labels_l, df)

if __name__ =="__main__":
    main()
