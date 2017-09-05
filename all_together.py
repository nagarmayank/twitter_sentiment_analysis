from capture_sentiment import capture_sentiment
from buildWordVector import buildWordVector
from clean_text import clean
from tweet_tokenize import tokenize
from labelizeTweets import labelizeTweets
from hashtags import gethashtags

import gensim
import json
from gensim import corpora
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

tweets_data = []
n_dim = 200
fname = r"C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\twitter_output.txt"
with open(fname) as f:
    data = f.read().splitlines()

for idx in range(len(data)):
    if data[idx] != '':
        all_data = json.loads(data[idx])
        tweets_data.append(all_data)

twitter_df = pd.DataFrame.from_dict(tweets_data)
print("Data loaded into DF")
twitter_df = twitter_df[twitter_df.lang=='en']
twitter_df = twitter_df[twitter_df.text.str.startswith('RT ')==False]
twitter_df = twitter_df[twitter_df.text.str.startswith('FAV ')==False]
twitter_df = twitter_df[twitter_df.text.str.contains('mPLUS')==False]
twitter_df = twitter_df.rename(columns = {'text':'SentimentText'})
twitter_df['hashtags'] = twitter_df.apply(gethashtags, axis=1)
print("Basic DF processing completed")

twitter_df['tokens'] = twitter_df['SentimentText'].map(tokenize)
print("tokenization is completed")

x_text = np.array(twitter_df.tokens)
x_text = labelizeTweets(x_text, 'ACTUAL')

tweet_vecs = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_text)])
tweet_vecs = scale(tweet_vecs)
print("word vectors are created")

df = capture_sentiment(twitter_df, tweet_vecs)

df_pos = df[df['Sentiment']==1]
df_neg = df[df['Sentiment']==0]

documents = list(df_pos['SentimentText'])
doc_clean = [clean(doc).split() for doc in documents]
print("documents are cleaned")

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)
dictionary.filter_extremes()

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
print("training LDA started")
ldamodel = Lda(doc_term_matrix,num_topics=5,id2word=dictionary,alpha=0.001,passes=100,eta=0.9)

for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words=10):
    print("Topic {}: Words: ".format(topic[0]))
    topicwords = [w for (w, val) in topic[1]]
    topicvalues = [val for (w, val) in topic[1]]
    print(topicwords)
#    print(topicvalues)
