import tensorflow as tf
import pandas as pd

# show tensorflow info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

## tf.compat.v1.estimator.LinearClassifier for estimator

# to show the transformation, no need to check the detail for now
# size is used for non-continuous features, the number of categorical column can be
def print_transformation(feature="age", continuous=True, size=2):
    # X = fc.numeric_column(feature)
    # Create feature name
    feature_names = [
        feature]

    # Create dict with the data
    d = dict(zip(feature_names, [df_train[feature]]))

    # Convert age
    if continuous == True:
        c = tf.feature_column.numeric_column(feature)
        feature_columns = [c]
    else:
        c = tf.feature_column.categorical_column_with_hash_bucket(feature, hash_bucket_size=size)
        c_indicator = tf.feature_column.indicator_column(c)
        feature_columns = [c_indicator]

    # Use input_layer to print the value
    input_layer = tf.feature_column.input_layer(
        features=d,
        feature_columns=feature_columns
    )
    # Create lookup table
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(input_layer, zero)
    # Return lookup table
    indices = tf.where(where)
    values = tf.gather_nd(input_layer, indices)
    # Initiate graph
    sess = tf.Session()
    # Print value
    print(sess.run(input_layer))


COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']

# -------------- Read data -----------------------
df_train = pd.read_csv("adult.data", skipinitialspace=True, names=COLUMNS, index_col=False)
df_test = pd.read_csv("adult.test", skiprows=1, skipinitialspace=True, names=COLUMNS, index_col=False)

## debug
#print(df_train.shape, df_test.shape)
#print(df_train.dtypes)

# convert object(string) to numeric value
label = {'<=50K' : 0, '>50K' : 1}
df_train.label = [label[i] for i in df_train.label]
label_test = {'<=50K.' : 0, '>50K.' : 1}
df_test.label = [label_test[i] for i in df_test.label]

## debug
#print(df_train["label"].value_counts())
#print(df_test["label"].value_counts())
#print(df_train.dtypes)

# -------------- Convert data -----------------------
# feature bucket
### Define continuous list
### convert to numeric value
CONT_FEATURES  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']

### Define the categorical list
CATE_FEATURES = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native_country']


# it has the same result as print_transformation()
continuous_features = [tf.feature_column.numeric_column(k) for k in CONT_FEATURES]
#print(continuous_features[0])

# a faster way to transform the data is to use the method categorical_column_with_hash_bucket.
# Altering string variables in a sparse matrix will be useful.
categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in CATE_FEATURES]
#print(categorical_features[0])

# debug to show the detail in a function
#print_transformation(feature="age", continuous=True)
#print_transformation(feature="workclass", continuous=False, size=5)

# -------------- Train classifier -----------------------

model = tf.estimator.LinearClassifier(
    n_classes = 2,
    model_dir = "train1",
    feature_columns = categorical_features+continuous_features
)


FEATURES = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation',
            'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country']
LABEL= 'label'


def get_input_fn(data_set, features, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in features}),
        y=pd.Series(data_set[LABEL].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


model.train(input_fn=get_input_fn(df_train,
                                   FEATURES,
                                   num_epochs=None,
                                   n_batch=128,
                                   shuffle=False),
            steps=1000)

# -------------- Evaluate it with test data -----------------------
model.evaluate(input_fn=get_input_fn(df_test,
                                     FEATURES,
                                     num_epochs=1,
                                     n_batch=128,
                                     shuffle=False),
               steps=1000)

# ============================ Improve evaluating data with polynomial regression ============================
# add a new column as age^2 named 'new', train and evaluate it again

def square_var(df_tr, df_te, var_name = 'age'):
    df_tr['new'] = df_tr[var_name].pow(2)
    df_te['new'] = df_te[var_name].pow(2)
    return df_tr, df_te


df_train_new, df_test_new = square_var(df_train, df_test, var_name='age')
print(df_train_new, df_test_new)

CONT_FEATURES_NEW  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week', 'new']
FEATURES_NEW = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_week', 'native_country', 'new']
continuous_features_new = [tf.feature_column.numeric_column(k) for k in CONT_FEATURES_NEW]
# NOTE: cannot train different models in the same directory
model_2 = tf.estimator.LinearClassifier(
    model_dir="train2",
    feature_columns=categorical_features+continuous_features_new
)
model_2.train(input_fn=get_input_fn(df_train_new,
                                    FEATURES_NEW,
                                    num_epochs=None,
                                    n_batch=128,
                                    shuffle=False),
              steps=1000)
model_2.evaluate(input_fn=get_input_fn(df_test_new,
                                       FEATURES_NEW,
                                       num_epochs=1,
                                       n_batch=128,
                                       shuffle=False),
                 steps=1000)

# ============================ Improve evaluating data with bucket and cross features ============================
# With these new features, the linear model can capture the relationship by learning different weights for each bucket.

age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

education_x_occupation = [tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000
)]
age_buckets_x_education_x_occupation = [tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
)]
base_columns = [
    age_buckets,
]
model_3 = tf.estimator.LinearClassifier(
    model_dir="train3",
    feature_columns=categorical_features+base_columns+education_x_occupation+age_buckets_x_education_x_occupation
)
FEATURES_3 = ['age','workclass', 'education', 'education_num', 'marital',
                'occupation', 'relationship', 'race', 'sex', 'native_country', 'new']

## todo: feature_columns in LinearClassifier() and FEATURES_3 are not matched??
model_3.train(input_fn=get_input_fn(df_train_new,
                                    FEATURES_3,
                                    num_epochs=None,
                                    n_batch=128,
                                    shuffle=False),
              steps=1000)
model_3.evaluate(input_fn=get_input_fn(df_test_new,
                                       FEATURES_3,
                                       num_epochs=1,
                                       n_batch=128,
                                       shuffle=False),
                 steps=1000)

# ============================ Improve evaluating data with regularization ============================
###
###  two regularization techniques:
#
#    L1: Lasso
#    L2: Ridge
# In TensorFlow, you can add these two hyperparameters in the optimizer.
# For instance, the higher the hyperparameter L2, the weight tends to be very low and close to zero.
# The fitted line will be very flat, while an L2 close to zero implies the weights are close to
# the regular linear regression.
#
model_4 = tf.estimator.LinearClassifier(
    model_dir="train4", feature_columns=categorical_features+base_columns+education_x_occupation+age_buckets_x_education_x_occupation,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.9,
        l2_regularization_strength=5))

model_4.train(input_fn=get_input_fn(df_train_new,
                                    FEATURES_3,
                                    num_epochs=None,
                                    n_batch=128,
                                    shuffle=False),
              steps=1000)
model_4.evaluate(input_fn=get_input_fn(df_test_new,
                                       FEATURES_3,
                                       num_epochs=1,
                                       n_batch=128,
                                       shuffle=False),
                 steps=1000)