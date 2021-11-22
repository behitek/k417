import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # used for plot interactive graph.
from wordcloud import WordCloud, STOPWORDS

sns.set_style('darkgrid')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('../tripadvisor_hotel_reviews.csv')

# Draw the rating distribution
rating = data['Rating'].values
sns.countplot(data=data, x="Rating", linewidth=3)
plt.title('Rating distribution', size=17)
plt.show()


# draw the word cloud

def get_words(rating=0):
    if rating > 0:
        content = '\n'.join(data[data['Rating'] == rating]['Review'].values.tolist())
    else:
        content = '\n'.join(data['Review'].values.tolist())
    content = re.sub('[^a-zA-Z]+', ' ', content)
    return content


stopwords = set(STOPWORDS)
content = get_words()
wordcloud = WordCloud(max_font_size=40, stopwords=stopwords).generate(content)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('WordCloud for all reviews', size=17)
plt.show()

for i in range(1, 6):
    content = get_words(i)
    wordcloud = WordCloud(max_font_size=40, stopwords=stopwords).generate(content)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title('WordCloud for reviews rating {}'.format(i), size=17)
    plt.show()
