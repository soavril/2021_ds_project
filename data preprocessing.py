import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

X = pd.read_csv('/Users/noah/desktop/code/2021_ds_project/wine_train.csv', index_col='id')
X_test_full = pd.read_csv('/Users/noah/desktop/code/2021_ds_project/wine_test.csv', index_col='id')

# training data의 형태는 다음과 같습니다.
X.head(5)

# predictor variable과 target variable을 분리합니다.
X= X.fillna('0')
X_test_full = X_test_full.fillna('0')
y = X['points']
X.drop(['points'], axis=1, inplace=True)

#title에서 year을 추출합니다.
X['year'] = X.title.str.extract('(\d+)')
X['year'] = X['year'].apply(str)
X_test_full['year'] = X_test_full.title.str.extract('(\d+)')
X_test_full['year'] = X_test_full['year'].apply(str)

#description_length를 추가합니다.
X = X.assign(description_length = X['description'].apply(len))
X_test_full = X_test_full.assign(description_length = X_test_full['description'].apply(len))

# Extract good and bad word
word_hig = pd.read_csv('/Users/noah/desktop/code/2021_ds_project/word_high.csv')
word_low = pd.read_csv('/Users/noah/desktop/code/2021_ds_project/word_low.csv')

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

X = X.assign(desc_good_word=np.nan)
X_test_full = X_test_full.assign(desc_good_word=np.nan)

extract_good_word(X)
extract_good_word(X_test_full)

def extract_bad_word(df):
    for i in df.index:
        desc = df.loc[i, 'description']
        score = 0
        for j in word_low.index:
            if word_low.loc[j, 'Word_low'] in desc:
                score += word_low.loc[j, 'low/high']
        df.loc[i, 'desc_bad_word'] = score

X = X.assign(desc_bad_word=np.nan)
X_test_full = X_test_full.assign(desc_bad_word=np.nan)

extract_bad_word(X)
extract_bad_word(X_test_full)

#상위에서 많이 나타나는 키워드 셋입니다. 
s = set(['impressive','velvety','beautifully','2018','beautiful','opulent','2020','2018','gorgeous','refined','cellar','delicious','powerful','2022','complex',
'2019','concentrated','polished','baking','2030','cru','lovely','six','graphite','years','develop','power','2025','least','brooding','site','wonderful','2025','decade',
'backbone','planted','compelling','potential','lavender','stunning','focused','star','mountain','elegant','fine-grained','vineyard','luscious','tightly','excellent'])

#description에서 s에 나타나는 단어의 빈도를 셉니다.
X_description_word = X['description'].copy()
X_description_copy = X['description'].copy()
for i in X_description_copy.index:
    X_description_word[i] = set(word_tokenize(X_description_copy[i]))
for i in X_description_word.index:
    X_description_word [i] = len(X_description_word[i] & s)
    
X_test_full_description_word = X_test_full['description'].copy()
X_test_full_description_copy = X_test_full['description'].copy()
for i in X_test_full_description_copy.index:
    X_test_full_description_word[i] = set(word_tokenize(X_test_full_description_copy[i]))
for i in X_test_full_description_word.index:
    X_test_full_description_word [i] = len(X_test_full_description_word[i] & s)
    
#description_point를 원래 데이터 프레임에 합쳐줍니다.
X_description_word = pd.DataFrame(X_description_word)
X_description_word.rename(columns = {'description' : 'description_point'}, inplace = True)
X = pd.concat([X, X_description_word], axis=1)

X_test_full_description_word = pd.DataFrame(X_test_full_description_word)
X_test_full_description_word.rename(columns = {'description' : 'description_point'}, inplace = True)
X_test_full = pd.concat([X_test_full, X_test_full_description_word], axis=1)

#텍스트에 대한 감성분석을 진행합니다.
desc_blob = [TextBlob(desc) for desc in X['description']]
X['tb_Pol'] = [b.sentiment.polarity for b in desc_blob]
X['tb_Subj'] = [b.sentiment.subjectivity for b in desc_blob]
analyzer = SentimentIntensityAnalyzer()
X['compound'] = [analyzer.polarity_scores(v)['compound'] for v in X['description']]
X['neg'] = [analyzer.polarity_scores(v)['neg'] for v in X['description']]
X['neu'] = [analyzer.polarity_scores(v)['neu'] for v in X['description']]
X['pos'] = [analyzer.polarity_scores(v)['pos'] for v in X['description']]

desc_blob = [TextBlob(desc) for desc in X_test_full['description']]
X_test_full['tb_Pol'] = [b.sentiment.polarity for b in desc_blob]
X_test_full['tb_Subj'] = [b.sentiment.subjectivity for b in desc_blob]
analyzer = SentimentIntensityAnalyzer()
X_test_full['compound'] = [analyzer.polarity_scores(v)['compound'] for v in X_test_full['description']]
X_test_full['neg'] = [analyzer.polarity_scores(v)['neg'] for v in X_test_full['description']]
X_test_full['neu'] = [analyzer.polarity_scores(v)['neu'] for v in X_test_full['description']]
X_test_full['pos'] = [analyzer.polarity_scores(v)['pos'] for v in X_test_full['description']]

print(X.head(5))