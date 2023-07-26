import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from app.input_func_maker import make_input_fun

# Load the titanic dataset
df_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
df_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique() # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

train_input_func = make_input_fun(df_train, y_train, num_epochs=20)
eval_input_func = make_input_fun(df_eval, y_eval, num_epochs=1, shuffle=False)

linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# train
linear_estimator.train(train_input_func)

# test
result = list(linear_estimator.predict(eval_input_func))
choice = int(input("Enter the person number : "))
print("Details of the person")
print(df_eval.loc[choice])
print(f"\nHis/Her predicted probability of surviving is {result[choice]['probabilities'][1]}")
print(f"Actual possibility of surviving is {y_eval.loc[choice]}")