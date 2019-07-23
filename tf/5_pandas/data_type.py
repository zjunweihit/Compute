#
#  Pandas uses
#    series for one-dimensional data structure and
#    DataFrame for multi-dimensional data structure
#
# A data frame is a two-dimensional array, with labeled axes (rows and columns)
# with rows to store the information and columns to name the information
#
# A series is a one-dimensional data structure.
#

import pandas as pd
import numpy as np

# Series

s_def_index = pd.Series([1, 2, 3])
s_abc_index = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

print("=================== Series ===================")
print("default index:\n{}\n".format(s_def_index))
print("charactor index:\n{}\n".format(s_abc_index))


# Data frame
# numpy array and data frame can be converted to each other

# numpy -> data frame
np_22 = [[1, 2], [3, 4]]
df_init_np = pd.DataFrame(np_22)

# data frame -> numpy
np_arr = np.array(df_init_np)

# data frame with dictionary
dict = { 'Name': ["Jerry", "Tom"], 'Age': [30, 40] }
df_init_dict = pd.DataFrame(dict)

print("=================== Data frame ===================")
print("data frame:\n{}\n".format(df_init_np))
print("numpy array:\n{}\n".format(np_arr))
print("data frame with dictionary:\n{}\n".format(df_init_dict))

# Range Data
#
# pd.data_range(start,period,frequency):
#   start: start date
#   frequency: day: 'D,' month: 'M' and year: 'Y.'


print("=================== Range Data ===================")
rd_day = pd.date_range('20190723', periods=7, freq='D')
print('Day:', rd_day)

rd_month = pd.date_range('20190723', periods=7, freq='M')
print('Month:', rd_month)


# Check Data
# describe: 25%, 50%, 75% means
#   min+(max-min)*percentage
# seems not working for random value

random_data = np.random.rand(7, 4)
df_init_rand = pd.DataFrame(random_data, index=rd_day, columns=list('ABCD'))

df_const = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7], [3, 2, 8, 9, 3, 1, 4]] , index=list('AB'), columns=rd_day)

print("=================== Check Data ===================")
print('data frame(random) all:\n{}\n'.format(df_init_rand))
print('data frame(random) head-3:\n{}\n'.format(df_init_rand.head(3)))
print('data frame(random) tail-2:\n{}\n'.format(df_init_rand.tail(2)))
print('data frame(random) summary:\n{}\n'.format(df_init_rand.describe()))

print('data frame(const) all:\n{}\n'.format(df_const))
print('data frame(const) summary:\n{}\n'.format(df_const.describe()))

# Slice Data
# get the column data by colum name

print("=================== Slice Data ===================")
print('get data frame(random) column(B):\n{}\n'.format(df_init_rand['B']))
print('get data frame(random) column(B, C):\n{}\n'.format(df_init_rand[['B', 'C']]))
print('get data frame(random) row(0:2):\n{}\n'.format(df_init_rand[0:2]))
print('get data frame(random) drop column(B, C):\n{}\n'.format(df_init_rand.drop(columns=['B', 'C'])))


# concatenate two Data frame
print("=================== concatenate 2 Data frame ===================")
con_df_1 = pd.DataFrame({'name': ['Tom', 'Jerry', 'Smith'],
                        'age': ['20', '30', '40']},
                        index=[0, 1, 2])
con_df_2 = pd.DataFrame({'name': ['Lucy', 'Jerry'],
                         'age': ['19', '10']},
                        index=[3, 4])
df_concat = pd.concat([con_df_1, con_df_2])
print('concatenate 2 data frame:\n{}\n'.format(df_concat))

# drop duplicates
print("=================== drop duplicates ===================")
print('concatenate 2 data frame(drop duplacate name):\n{}\n'.format(df_concat.drop_duplicates('name')))

# sort age
print("=================== sort age ===================")
print('concatenate 2 data frame(sort age):\n{}\n'.format(df_concat.sort_values('age')))

# rename: change index
print("=================== change index ===================")
print('concatenate 2 data frame(change index):\n{}\n'.format(df_concat.rename(columns={'name':'surname', 'age': 'age_ppl'})))
