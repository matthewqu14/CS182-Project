#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:39:55 2020

@author: alfiantjandra
"""
# Python program to generate WordCloud 
  
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from PIL import Image
 
#Generate mask from Prof.Boaz, Prof.Milind image
boaz_mask = np.array(Image.open("boaz.png"))
milind_mask = np.array(Image.open("milind.png"))
 
# Reads text file 
df = pd.read_csv("all_depressed_tweets.csv", encoding ="latin-1") 
  
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df['tweet']: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "


'''Choose between "mask=boaz_mask" or "mask=milind_mask" '''
wordcloud = WordCloud(width = 800, height = 800, max_words= 100,
                      
                      
                mask = milind_mask
                
                ,contour_width = 1, background_color ='white', 
                colormap = 'Blues', stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image      
                  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation  = 'bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 
# plt.savefig('wordcloud.png')
plt.show()


# background = Image.open("boaz_background.png")
# foreground = Image.open("wordcloud.png")

# background.paste(foreground, (0, 0), foreground)
# background.show()