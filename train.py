from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

# 모델을 만들고 training 시킵니다. catboost모델을 사용합니다.

model = CatBoostRegressor()

def train(X_train, X_valid, y_train, y_valid, categorical_columns):
    model.fit(X_train, y_train, categorical_columns, plot = True)
    valid_preds = model.predict(X_valid)
    print('RMSE: {}'.format(mean_squared_error(y_valid, valid_preds, squared=False)))

# test data에 대해서도 같은 preprocessing 작업을 해준 후, 훈련된 모델을 이용해 결과를 예측합니다.
def test(X_test):
    test_preds = model.predict(X_test)
    return test_preds

# Submission 파일을 만듭니다.
my_submission = pd.DataFrame({'id': X_test.index, 'points': test_preds})
my_submission.to_csv('wine_my_submission.csv', index=False)