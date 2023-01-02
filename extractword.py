import pandas as pd 


word_hig = pd.read_csv('./word_high.csv')
word_low = pd.read_csv('./word_low.csv')

word_hig = word_hig[['Word_high', 'high/low']]
word_low = word_low[['Word_low', 'low/high']]

def extract_good_word(df):
    for i in df.index:
        desc = df.loc[i, 'description']
        score = 0
        for j in word_hig.index:
            if word_hig.loc[j, 'Word_high'] in desc:
                score += word_hig.loc[j, 'high/low']
        df.loc[i, 'desc_good_word'] = score

def extract_bad_word(df):
    for i in df.index:
        desc = df.loc[i, 'description']
        score = 0
        for j in word_low.index:
            if word_low.loc[j, 'Word_low'] in desc:
                score += word_low.loc[j, 'low/high']
        df.loc[i, 'desc_bad_word'] = score



