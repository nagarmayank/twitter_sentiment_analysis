import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = "%s_%s" %(label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized