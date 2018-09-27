from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


text = open('data/12345.txt').read()
alice_coloring = np.array(Image.open("data/12345.png"))
stopwords = set(STOPWORDS)
wc = WordCloud(font_path="C:/Windows/Fonts/STFANGSO.ttf", background_color="white", max_words=2000, mask=alice_coloring,
               stopwords=stopwords, max_font_size=40, random_state=42)
wc.generate(text)
image_colors = ImageColorGenerator(alice_coloring)
# 在只设置mask的情况下,你将会得到一个拥有图片形状的词云
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis("off")
plt.show()