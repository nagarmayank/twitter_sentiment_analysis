hash = []
def gethashtags(row):
    tags = []
    ls = row['entities']['hashtags']
    for i in range(len(ls)):
        hash.append(ls[i]['text'])
        tags.append(ls[i]['text'])
    return tags