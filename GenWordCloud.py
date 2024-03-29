
import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import random
import pi
import csv

from wordcloud import WordCloud, STOPWORDS


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

d = path.dirname(__file__)

mask = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))
text = open("a_new_hope.txt").read()

stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

wc = WordCloud(max_words=2000, mask=mask, stopwords=stopwords, margin=10,
               random_state=1).generate(text)
default_colors = wc.to_array()
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
wc.to_file("a_new_hope.png")
plt.axis("off")
plt.figure()
plt.title("Trump vs Hillary - 2016/10/20")
plt.imshow(default_colors)
plt.axis("off")
plt.show()
