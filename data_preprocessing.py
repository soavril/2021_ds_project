import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from extractword import *

X = pd.read_csv('./wine_train.csv', index_col='id')
X_test_full = pd.read_csv('./wine_test.csv', index_col='id')

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
X = X.assign(desc_good_word=np.nan)
X_test_full = X_test_full.assign(desc_good_word=np.nan)

X = X.assign(desc_bad_word=np.nan)
X_test_full = X_test_full.assign(desc_bad_word=np.nan)

extract_good_word(X)
extract_good_word(X_test_full)

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

# training data에서 validation set을 나눠줍니다.
 
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#feature column과 categorical column을 구분해줍니다.

feature_columns = ['country', 'description', 'designation', 'price', 'province',
       'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
       'variety', 'winery', 'year', 'description_length', 'desc_good_word', 'desc_bad_word', 'tb_Pol', 'tb_Subj', 'compound', 'neg', 'neu', 'pos']

X_train = X_train_full[feature_columns].copy()
X_valid = X_valid_full[feature_columns].copy()
X_test = X_test_full[feature_columns].copy()

categorical_columns = ['country', 'description', 'designation', 'province',
       'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
       'variety', 'winery', 'year']