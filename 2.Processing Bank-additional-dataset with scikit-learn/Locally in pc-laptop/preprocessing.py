## Single command-line parameter with the argparse library 
## the ratio for the training and test datasets.
## The actual value will be passed to the script by the SageMaker Processing SDK:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train-test-split-ratio',type=float, default=0.3)
args, _ = parser.parse_known_args()
print('Received arguments {}'.format(args))
split_ratio = args.train_test_split_ratio

# We load the input dataset using pandas . At startup, 
# SageMaker Processing automatically copied it from S3 to a user-defined location
# inside the container; here, it is 
# /opt/ml/processing/input :

import os
import pandas as pd
input_data_path = os.path.join('/opt/ml/processing/input','bank-additional-full.csv')
df = pd.read_csv(input_data_path)#,sep =";")

# Then, we remove any line with missing values, as well as duplicate lines:
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#Then, we count negative and positive samples, and display the class ratio.
#This will tell us how unbalanced the dataset is :

one_class = df[df['y']=='yes']
one_class_count = one_class.shape[0]
zero_class = df[df['y']=='no']
zero_class_count = zero_class.shape[0]
zero_to_one_ratio = zero_class_count/one_class_count
print("Ratio: %.2f" % zero_to_one_ratio)

import numpy as np
df['no_previous_contact'] = np.where(df['pdays'] == 999, 1, 0)

df['not_working'] = np.where(np.in1d(df['job'], ['student', 'retired', 'unemployed']), 1, 0)

# Now, let's split the dataset into training and test sets. Scikit-learn has a convenient
# API for this, and we set the split ratio according to a command-line argument
# passed to the script:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1),df['y'],
                                                    test_size=split_ratio, random_state=0)
                                                    
# The next step is to scale numerical features and to one-hot encode the categorical
# features. We'll use StandardScaler for the former, and OneHotEncoder for
# the latter:
    
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

columns_to_scale = ['age', 'duration', 'campaign', 'pdays', 'previous']
columns_to_encode=['job', 'marital', 'education', 'default', 'housing','loan','contact', 'month', 'day_of_week','poutcome']


# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

p = Pipeline(
    [("coltransformer", ColumnTransformer(
        transformers=[
            ("assessments", Pipeline([("scale", scaler)]), columns_to_scale),
            ("ranks", Pipeline([("encode", ohe)]), columns_to_encode),
        ]),
    )]
)

# Then, we process the training and test datasets:

train_features = p.fit_transform(X_train)
test_features = p.transform(X_test)

# Finally, we save the processed datasets, separating the features and labels.
# They're saved to user-defined locations in the container, and SageMaker Processing
# will automatically copy the files to S3 before terminating the job:

train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')

pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

y_train.to_csv(train_labels_output_path, header=False, index=False)
y_test.to_csv(test_labels_output_path, header=False, index=False)