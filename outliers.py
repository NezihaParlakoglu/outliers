# functional data preprocessing
# function for outliers thresholds

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv('C:/Users/Neziha/PycharmProjects/6_Feature_Engineering/dataset/titanic.csv')
df = df_.copy()
df.head()


def outliers_thresholds(dataframe,col_name,q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return  low_limit, up_limit


outliers_thresholds(df,"Age")
outliers_thresholds(df,"Fare")
outliers_thresholds(df,"Survived")

# Is there outliers?

def check_outlier(dataframe,col_name):
    low_limit, up_limit = outliers_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df,"Age")
check_outlier(df,"Fare")
check_outlier(df,"Survived")


# grab col names (categoric-numeric)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    bring the names os numeric, categoric and categoric but cardinal variables
    Notes: categoric variables which is  seen as numeric is included in categoric group

    Parametres;
    -------
        dataframe: dataframe
        cat_th:int, optional
                threshould value for categoric class
        car_th:int, optional
                threshould value for categoric class but cardinal

    Returns
    -------
        cat_cols:List
            list of categoric column names
        num_cols:List
            list of categoric column names
    """

    # cat_cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col  not in cat_but_car ]

    # num cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


grab_col_names(df)

# reach outliers

def grab_outliers(dataframe, col_name, index=True):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    if check_outlier(dataframe, col_name):
        if index:
            return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].index
        else:
            if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].shape[0] > 10:
                return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].head()
            else:
                return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    else:
        print("No outliers")


grab_outliers(df,"Fare",index=False)
grab_outliers(df,"Survived",index=False)

# remove outliers

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.head(20)

for col in num_cols:
    new_df = remove_outlier(df, col )


new_df.shape

df.shape[0] - new_df.shape[0]

# re-assignment with thresholds

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit


for col in num_cols:
    print(col, check_outlier(df,col))


for col in num_cols:
        replace_with_thresholds(df,col)



###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")




#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor  (LOF yönteminde komşu uzaklık yoğunluklarına bakılır)
#############################################
# 17 years old with 3 mariage


