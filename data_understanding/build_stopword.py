import re

import pandas as pd
import seaborn as sns  # used for plot interactive graph.

sns.set_style('darkgrid')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('../tripadvisor_hotel_reviews.csv')


def get_words(rating=0):
    if rating > 0:
        content = '\n'.join(data[data['Rating'] == rating]['Review'].values.tolist())
    else:
        content = '\n'.join(data['Review'].values.tolist())
    content = re.sub('[^a-zA-Z]+', ' ', content)
    words = content.split()
    return words


all_unique_words = set(get_words())
one_star_words = get_words(1)
two_star_words = get_words(2)
three_star_words = get_words(3)
four_star_words = get_words(4)
five_star_words = get_words(5)

count_1_star = {}
count_2_stars = {}
count_3_stars = {}
count_4_stars = {}
count_5_stars = {}

for word in one_star_words:
    count_1_star[word] = count_1_star.get(word, 0) + 1
for word in two_star_words:
    count_2_stars[word] = count_2_stars.get(word, 0) + 1
for word in three_star_words:
    count_3_stars[word] = count_3_stars.get(word, 0) + 1
for word in four_star_words:
    count_4_stars[word] = count_4_stars.get(word, 0) + 1
for word in five_star_words:
    count_5_stars[word] = count_5_stars.get(word, 0) + 1

count = {}
for word in all_unique_words:
    min_val = min(count_1_star.get(word, 0),
                  count_2_stars.get(word, 0),
                  count_3_stars.get(word, 0),
                  count_4_stars.get(word, 0),
                  count_5_stars.get(word, 0))
    count[word] = min_val

with open('stopwords.txt', 'w') as fp:
    for word in sorted(count, key=count.get, reverse=True)[:100]:
        fp.write(word + '\n')
        print(word, count[word])
