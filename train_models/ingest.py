import pandas as pd

def ingest(file):
    df = pd.read_csv(file, error_bad_lines=False)
    df.drop(['ItemID','SentimentSource'], axis=1, inplace=True)
    df = df.where(df.Sentiment.isnull() == False).dropna()
    df['Sentiment'] = df['Sentiment'].map(int)
    df = df.where(df.SentimentText.isnull() == False).dropna()
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    print(df.shape)
    return df