import pandas as pd
from sklearn import preprocessing, grid_search
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

py.sign_in('eddyrain', 'jld56fU3vKxm01UCcuHM')


def read_file(file):
    df = pd.read_csv(file, sep=';')
    return df


def encode(x): return 1 if x in [46] else 0  # The house is in the #46 district or not


def predict_null(df):
    process_df = df[
        ['SIZE_LIVING', 'SOLD_PRICE_EUR', 'SIZE_FULL', 'FLOORS', 'GARAGE', 'ROOMS', 'BATHROOMS', 'SIZE_ABOVE',
         'SIZE_BASEMENT', 'CONDITION_GRADE', 'DISTRICT', 'RENOVATION_DATE', 'BUILT_DATE', 'SOLD_DATE']]

    known = process_df[process_df['SIZE_LIVING'].notnull()].as_matrix()
    unknown = process_df[process_df['SIZE_LIVING'].isnull()].as_matrix()

    X = known[:, 1:]  # Features
    Y = known[:, 0]  # label

    rfr = RandomForestRegressor(random_state=0, n_estimators=1500, n_jobs=-1)
    rfr.fit(X, Y)

    predicted = rfr.predict(unknown[:, 1::])
    df.loc[(df['SIZE_LIVING'].isnull()), 'SIZE_LIVING'] = predicted

    return df


def fill_median(df, target):
    median = df[target].median()
    df[target] = df[target].fillna(median)  # Median
    return df


# Change the data type of DISTRICT
def transform_roman_num2_alabo(df):
    define_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    for index, row in df.iterrows():
        if row['DISTRICT'] == '0':
            return 0
        else:
            res = 0
            for i in range(0, len(row['DISTRICT'])):
                if i == 0 or define_dict[row['DISTRICT'][i]] <= define_dict[row['DISTRICT'][i - 1]]:
                    res += define_dict[row['DISTRICT'][i]]
                else:
                    res += define_dict[row['DISTRICT'][i]] - 2 * define_dict[row['DISTRICT'][i - 1]]
            df.at[index, 'DISTRICT'] = res
    return df


def save_excel(df, name):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    filename = name + '.xlsx'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()


def train_model(df, cols, input):
    df = df[cols]
    cols = cols
    cols.insert(0, cols.pop(cols.index('SOLD_PRICE_EUR')))
    df = df.ix[:, cols]
    print("This is df: ", df.loc[df['BATHROOMS'] == 2])
    # x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    x, y = df.iloc[:, 1:], df.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    parameters_rfr = [{'n_estimators': [1500],
                       'max_depth': [5, 10],
                       'min_samples_split': [2, 5],
                       'min_weight_fraction_leaf': [0.0, 0.1, 0.2]
                       }]
    rfr = GridSearchCV(RandomForestRegressor(), parameters_rfr)
    rfr.fit(x_train, y_train)
    print("This is score: ", rfr.best_score_)
    print("This is best parameters: ", rfr.best_params_)
    predictions_rfr = rfr.predict(x_test)
    # for i in range(len(predictions_rfr)):
    #     print("This is real: ", y_test.values[i])
    #     print("This is prediction: ", predictions_rfr[i] )
    print('RMSE is: \n', mean_squared_error(y_test, predictions_rfr))
    result_rfr = rfr.predict(input)
    print("RFR is DONE!")

    "Normaliztion step"
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    input = scaler.transform(input)

    parameters_test2 = {'fit_intercept': [True, False]}
    regr_lr = GridSearchCV(estimator=LinearRegression(), param_grid=parameters_test2, cv=3)
    regr_lr.fit(x_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (regr_lr.best_params_, regr_lr.best_score_))

    predictions_lr = regr_lr.predict(x_test)
    # for i in range(len(predictions_lr)):
    #     print("This is real: ", y_test.values[i])
    #     print("This is prediction: ", predictions_lr[i])
    print('RMSE is: \n', mean_squared_error(y_test, predictions_lr))
    result_lr = regr_lr.predict(input)
    print("LR is DONE!")

    C_range = np.logspace(-1, 6, num=5)
    gamma_range = np.logspace(-4, 1, num=5)
    svr_rbf = GridSearchCV(SVR(kernel='rbf'), cv=3,
                           param_grid={"C": C_range,
                                       "gamma": gamma_range}, n_jobs=1)
    svr_rbf.fit(x_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (svr_rbf.best_params_, svr_rbf.best_score_))
    predictions_svr = svr_rbf.predict(x_test)
    # for i in range(len(predictions_svr)):
    #     print("This is real: ", y_test.values[i])
    #     print("This is prediction: ", predictions_svr[i])
    print('RMSE is: \n', mean_squared_error(y_test, predictions_svr))
    result_svr = svr_rbf.predict(input)
    print("SVR is DONE!")

    return result_rfr, result_lr, result_svr


def plot_bar(df, target):
    sns.set_style("whitegrid")
    ax = sns.barplot(x=target, y="SOLD_PRICE_EUR", data=df)
    plt.xticks(rotation=0)
    plt.show()
    fig = ax.get_figure()
    filename = "SOLD_PRICE_EUR_VS_" + target + "_bar.png"
    fig.savefig(filename)


def plot_pie(df):
    lables_1 = df.dtypes.value_counts().keys().astype(str).tolist()
    values_1 = df.dtypes.value_counts().tolist()
    lables_2 = ['Not duplicated', 'Duplicated']
    temp_1 = df[df.duplicated() == False]
    temp_2 = df[df.duplicated() == True]
    values_2 = [temp_1.shape[0], temp_2.shape[0]]

    trace1 = {
        "domain": {
            "x": [0, 0.48],
            "y": [0, 0.49]
        },
        "labels": lables_1,
        "marker": {"colors": ["rgb(33, 75, 99)", "rgb(79, 129, 102)", "rgb(151, 179, 100)", "rgb(175, 49, 35)",
                              "rgb(36, 73, 147)"]},
        "name": "Starry Night",
        "textinfo": "none",
        "type": "pie",
        "values": values_1
    }
    trace2 = {
        "domain": {
            "x": [0.52, 1],
            "y": [0, 0.49]
        },
        "labels": lables_2,
        "marker": {"colors": ["rgb(146, 123, 21)", "rgb(177, 180, 34)", "rgb(206, 206, 40)", "rgb(175, 51, 21)",
                              "rgb(35, 36, 21)"]},
        "name": "Sunflowers",
        "textinfo": "none",
        "type": "pie",
        "values": values_2
    }
    # trace3 = {
    #     "domain": {
    #         "x": [0, 0.48],
    #         "y": [0.51, 1]
    #     },
    #     "hoverinfo": "label+percent+name",
    #     "labels": ["1st", "2nd", "3rd", "4th", "5th"],
    #     "marker": {"colors": ["rgb(33, 75, 99)", "rgb(79, 129, 102)", "rgb(151, 179, 100)", "rgb(175, 49, 35)",
    #                           "rgb(36, 73, 147)"]},
    #     "name": "Irises",
    #     "textinfo": "none",
    #     "type": "pie",
    #     "values": [38, 19, 16, 14, 13]
    # }
    # trace4 = {
    #     "domain": {
    #         "x": [0.52, 1],
    #         "y": [0.51, 1]
    #     },
    #     "hoverinfo": "label+percent+name",
    #     "labels": ["1st", "2nd", "3rd", "4th", "5th"],
    #     "marker": {"colors": ["rgb(146, 123, 21)", "rgb(177, 180, 34)", "rgb(206, 206, 40)", "rgb(175, 51, 21)",
    #                           "rgb(35, 36, 21)"]},
    #     "textinfo": "none",
    #     "type": "pie",
    #     "values": [31, 24, 19, 18, 8]
    # }
    data = Data([trace1, trace2])
    layout = {
        "showlegend": True,
        "title": "Data Management_Nordea case"
    }
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)


def box_plot(df, target):
    sns.set(font_scale=0.8)
    var = target
    data = pd.concat([df['SOLD_PRICE_EUR'], df[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SOLD_PRICE_EUR", data=data, showmeans=True)
    fig.axis(ymin=0, ymax=2500000)
    plt.xticks(rotation=90)
    # plt.xticks(rotation=0)
    plt.show()
    fig = fig.get_figure()
    filename = "SOLD_PRICE_EUR_VS_" + target + "_box.png"
    fig.savefig(filename)


def plot_scatter(df, target):
    sns.set(font_scale=0.7)
    var = target
    data = pd.concat([df['SOLD_PRICE_EUR'], df[var]], axis=1)
    fig = sns.regplot(x=var, y="SOLD_PRICE_EUR", data=data)
    filename = "SOLD_PRICE_EUR_VS_" + var + "_scatter.png"
    plt.show()
    fig = fig.get_figure()
    fig.savefig(filename)


def plot_outlier(df, col):
    sns.set()
    cols = col
    sns.pairplot(df[cols], size=2.5)
    plt.show()


def plot_correlation_matrix(df):
    col = ['SOLD_PRICE_EUR', 'ROOMS', 'BATHROOMS', 'SIZE_LIVING', 'SIZE_FULL', 'SIZE_ABOVE', 'SIZE_BASEMENT']
    df = df[col]
    corrmat = df.corr()
    k = 7  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SOLD_PRICE_EUR')['SOLD_PRICE_EUR'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=0.7)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 9},
                     yticklabels=cols.values, xticklabels=cols.values, cmap="YlGnBu")
    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    fig = hm.get_figure()
    filename = "Corr_matrix_SOLD_PRICE_EUR_numerical"
    fig.savefig(filename)


def check_corr(df):
    numeric_features = df.select_dtypes(include=[np.number])
    corr = numeric_features.corr()
    print(corr['SOLD_PRICE_EUR'].sort_values(ascending=False)[:5], '\n')
    print(corr['SOLD_PRICE_EUR'].sort_values(ascending=False)[-5:], '\n')


def check_outliers(df, col):
    sns.set(color_codes=True, font_scale=0.8)

    var = col
    data = pd.concat([df['SOLD_PRICE_EUR'], df[var]], axis=1)
    fig = sns.regplot(x=var, y='SOLD_PRICE_EUR', data=data)
    filename = "SOLD_PRICE_EUR_VS_" + var + ".png"
    plt.show()
    fig = fig.get_figure()
    fig.savefig(filename)


def check_duplicted(df):
    duplicated_row = df[df.duplicated() == True]
    # print("This is duplicated row: ", duplicated_row)
    return df.drop_duplicates()


def check_null(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("This is : ", missing_data.head(15))
    return missing_data


def handel_null(df, method):
    colunm_header = list(df)
    columns = ['Name', '# Not null', '# Null']
    df_new = pd.DataFrame(columns=columns)
    size_df = df.shape[0]
    # for index, value in enumerate(colunm_header):  # Get the number of null and not_null
    #     df_new.at[index, 'Name'] = value
    #     df_new.at[index, '# Not null'] = df[value].notnull().value_counts().tolist()[0]
    #     df_new.at[index, '# Null'] = size_df - df[value].notnull().value_counts().tolist()[0]

    """
    Check where the null comes from (SIZE_LIVING),
    Is it important for prediction? 
    More than 15% delete the feature
    """
    """
    Handel the missing value:
    1: fillna(A number)
    2: dropna() For BUILT_DATE and SOLD_PRICE_EUR Because only 3 records are missing
    3: Fill in by prediction model. For SIZE_LIVING, because 10% records are missing and this feature is important
    """
    # save_excel(df_new,'number_of_null')
    # plot_bar(df_new)
    # check_null(df)

    df = df.drop(df.loc[df['BUILT_DATE'].isnull()].index)
    df = df.drop(df.loc[df['SOLD_PRICE_EUR'].isnull()].index)
    df['BUILT_DATE'] = df['BUILT_DATE'].astype(np.int64)
    # Only SIZE_LIVING has null from now on
    # print("This is size of df: ", df.shape)
    if method == 1:  # drop null
        df = df.dropna()
    elif method == 2:  # Using median to fill in the null
        df = fill_median(df, 'SIZE_LIVING')
    elif method == 3:
        df = predict_null(df)  # 'SIZE_LIVING'

    # print("This is : ", df.describe())
    # plot_scatter(df,'BUILT_DATE')
    # box_plot(df,'DISTRICT')
    # missing_data = check_null(df)
    # box_plot(df, 'BUILT_DATE') # the relationship with SOLD_PRICE_EUR
    # box_plot(df,'RENOVATION_DATE')
    # box_plot(df,'CONDITION_GRADE')
    # print("This is df: ",df.describe().astype(np.int64).T)
    return df


def main():
    file = 'House Prices Case Study.csv'
    input = [2, 150]  # 150 square meter living space and 2 bathrooms
    # input = [2,150,1] # 150 square meter living space and 2 bathrooms with a garage
    # input = [2,150,1,2] # 150 square meter living space and 2 bathrooms with a garage, 2 rooms
    input = np.asarray(input).reshape(1, -1)

    df = read_file(file)
    # print("This is df: ", df.describe())
    # plot_pie(df) # plot the pie in order to know the data type and duplicated data
    df = check_duplicted(df)  # drop the duplicated record
    # change the data type of FLOORS
    df['FLOORS'] = df['FLOORS'].apply(lambda x: pd.to_numeric(x.replace(',', '.'), errors='coerce'))
    df = transform_roman_num2_alabo(df)  # 'DISTRICT'
    df['DISTRICT'] = df['DISTRICT'].astype(np.int64)
    df['FLOORS'] = df['FLOORS'].astype(np.float)

    # sns.set(color_codes=True, font_scale=0.8)
    # fig = sns.jointplot(x="SIZE_LIVING", y="SOLD_PRICE_EUR", data=df, kind='reg', size=5)
    # plt.show()
    # filename = "SOLD_PRICE_EUR_IMPACTED_BY_SIZE_LIVING"  # Sold time is useful or not?
    # fig.savefig(filename)
    # numeric_features = df.select_dtypes(include=[np.number])
    # numeric_features.dtypes
    # corr = numeric_features.corr()
    # print(corr['SOLD_PRICE_EUR'].sort_values(ascending=False)[:7], '\n')
    # print(corr['SOLD_PRICE_EUR'].sort_values(ascending=False)[-7:], '\n')

    # plot_bar(df, 'ROOMS')
    # plot_bar(df, 'GARAGE')
    # plot_bar(df,'DISTRICT')
    # plot_scatter(df, 'SIZE_LIVING')
    # plot_scatter(df, 'SIZE_ABOVE')
    # plot_scatter(df, 'SIZE_BASEMENT')
    # box_plot(df, 'DISTRICT') # if people live in the District 46, the price should be higher
    # box_plot(df, 'FLOORS')
    # plot_bar(df,'FLOORS')

    """
    If the appartment is in the good district (#46) or not?
    """
    # df['DISTRICT'] = df.DISTRICT.apply(encode)
    # plot_bar(df,'DISTRICT')

    """
    Handel the null
    """
    df = handel_null(df, method=3)  # solve the problem of null, 3 means using imputation method

    """
    Handel the outliers
    """
    # temp = df.sort_values(by='SIZE_LIVING', ascending=False)[:-1]  # Find the outlier ID and delete it

    # df = df.drop(df[df['UNIQUE_ID'] == 1458428820].index)
    df = df.drop(df[df['UNIQUE_ID'] == 1721775748].index)
    df = df.drop(df[df['UNIQUE_ID'] == 1112178615].index)
    # df = df.drop(df[df['UNIQUE_ID'] == 1495181813].index)

    """
    Select the features:
    
    Calculating the relationship between numerical variables and continuous variable (SOLD_PRICE_EUR), we can use
    correlation matrix.
    
    Calculating the relationship between a binary (0,1) and continuous variable (SOLD_PRICE_EUR), we can use
    Point biserial correlation r.
    
    Calculating the relationship between categorical and continuous variable (SOLD_PRICE_EUR), we can draw the 
    plot which has a linear regression line.

    """

    # fig =sns.jointplot(x="DISTRICT", y="SOLD_PRICE_EUR", data=df, kind='reg', size=5)
    # box_plot(df,'DISTRICT')
    # plt.show()
    # filename = "SOLD_PRICE_EUR_IMPACTED_BY_DISTRICT_reg_binary.png"  # Sold time is useful or not?
    # fig.savefig(filename)
    # box_plot(df,'GARAGE')
    # r, p = stats.pointbiserialr(df['GARAGE'], df['SOLD_PRICE_EUR'])
    # print('GARAGE point biserial correlation r is %s with p = %s' % (r, p))  # 0.2669
    # r, p = stats.spearmanr(df['CONDITION_GRADE'],
    #                        df['SOLD_PRICE_EUR'])
    # print('point biserial correlation r is %s with p = %s' % (r, p)) # 0.0166

    """
    Does SOLD_DATE,RENOVATION_DATE and BUILD_DATE important?
    """

    df['AGE'] = df.apply(lambda rec: int(str(rec.SOLD_DATE)[0:4]) - rec.BUILT_DATE, axis=1)
    # box_plot(df,'AGE')
    df['IS_REV'] = df['RENOVATION_DATE'].map(lambda rec: int(rec != 0))
    df['REV_YEAR'] = df.apply(lambda rec: int(int(str(rec.SOLD_DATE)[0:4]) - rec.RENOVATION_DATE), axis=1)
    # box_plot(df, 'REV_YEAR')
    # box_plot(df, 'IS_REV')
    # r, p = stats.spearmanr(df['IS_REV'],
    #                        df['SOLD_PRICE_EUR'])
    # print('IS_REV point biserial correlation r is %s with p = %s' % (r, p))

    df['SOLD_YEAR'] = df['SOLD_DATE'].map(lambda rec: int(str(rec)[0:4]))
    df['SOLD_MONTH'] = df['SOLD_DATE'].map(lambda rec: int(str(rec)[4:6]))

    # box_plot(df, 'REV_YEAR')
    # box_plot(df, 'SOLD_YEAR')
    # box_plot(df,'SOLD_MONTH')
    # r, p = stats.spearmanr(df['SOLD_YEAR'],
    #                        df['SOLD_PRICE_EUR'])
    # print('point biserial correlation r is %s with p = %s' % (r, p))

    df = df.drop(['UNIQUE_ID'], axis=1)
    # plot_correlation_matrix(df)

    """
    Building model and prediction by using selected features
    """
    # col = ['BATHROOMS', 'SIZE_LIVING',
    #        'GARAGE', 'SOLD_PRICE_EUR']
    # col = ['BATHROOMS', 'SIZE_LIVING',
    #        'GARAGE', 'ROOMS', 'SOLD_PRICE_EUR']
    col = ['BATHROOMS', 'SIZE_LIVING',
           'SOLD_PRICE_EUR']
    rfr, liner_reg, svr = train_model(df, col, input)

    print("This is prediction linear: ", liner_reg)
    print("This is prediction RF: ", rfr)
    print("This is prediction SVR: ", svr)
    print("This is average price: ", int((rfr + svr) / 2))

    # sns.set_style("whitegrid")
    # y = [765783,796759,403947]
    # x = [4,3,2]
    # fig = sns.barplot(x,y)
    #
    # filename = "Prediction based on different features.png"
    # plt.show()
    # fig = fig.get_figure()
    # fig.savefig(filename)



if __name__ == '__main__':
    main()
