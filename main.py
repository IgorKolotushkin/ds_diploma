import joblib
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def delete_columns(df):
    df_copy = df.copy()
    columns_to_drop = [
        'utm_campaign',
        'utm_adcontent',
        'utm_keyword',
    ]

    return df_copy.drop(columns_to_drop, axis=1)


def add_new_features(df):
    df_copy = df.copy()

    df_copy['screen_square'] = df_copy['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1])
    )

    df_copy['month'] = df_copy['visit_date'].apply(lambda x: int(x.split('-')[1]))
    df_copy['day'] = df_copy['visit_date'].apply(lambda x: int(x.split('-')[2]))
    df_copy['hour'] = df_copy['visit_time'].apply(lambda x: int(x.split(':')[0]))

    return df_copy


def delete_outliers(df):
    df_copy = df.copy()

    def calculate_outliers(data):
        q25 = data.screen_square.quantile(0.25)
        q75 = data.screen_square.quantile(0.75)
        iqr = q75 - q25

        return q25 - 1.5 * iqr, q75 + 1.5 * iqr

    boundaries = calculate_outliers(df_copy)
    df_copy.loc[df_copy.screen_square < boundaries[0], 'screen_square'] = int(boundaries[0])
    df_copy.loc[df_copy.screen_square > boundaries[1], 'screen_square'] = int(boundaries[1])

    return df_copy


def categorical_inputer(df):
    df_copy = df.copy()

    device_list = [
        'mobile',
        'tablet',
    ]
    df_copy.loc[
        (df_copy['device_brand'] == 'Huawei') & (df_copy['device_category'] == 'desktop'), 'device_category'
    ] = 'mobile'
    df_copy.loc[
        (df_copy['device_brand'] == 'Samsung') & (df_copy['device_category'] == 'desktop'), 'device_category'
    ] = 'mobile'
    for device in device_list:
        df_copy.loc[
            (df_copy['device_brand'] == 'Xiaomi') & (df_copy['device_os'].isna()) & (df_copy['device_category'] == device), 'device_os'
        ] = 'Android'
        df_copy.loc[
            (df_copy['device_brand'] == 'Huawei') & (df_copy['device_os'].isna()) & (df_copy['device_category'] == device), 'device_os'
        ] = 'Android'
        df_copy.loc[
            (df_copy['device_brand'] == 'Samsung') & (df_copy['device_os'].isna()) & (df_copy['device_category'] == device), 'device_os'
        ] = 'Android'
    df_copy.loc[
        (df_copy['device_brand'] == 'Apple') & (df_copy['device_category'].isin(['mobile', 'tablet'])) & (df_copy['device_os'].isna()), 'device_os'
    ] = 'iOS'
    df_copy.loc[
        (df_copy['device_brand'] == 'Apple') & (df_copy['device_category'] == 'desktop') & (df_copy['device_os'].isna()), 'device_os'
    ] = 'Macintosh'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == 'Windows'), 'device_brand'
    ] = 'other_brand'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'].isna()) & (df_copy['device_category'] == 'desktop'), 'device_os'
    ] = 'Windows'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == 'Macintosh'), 'device_brand'
    ] = 'Apple'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == '(not set)') & (df_copy['device_category'] == 'desktop'), 'device_brand'
    ] = 'other_brand'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == 'Chrome OS') & (df_copy['device_category'] == 'desktop'), 'device_brand'
    ] = 'other_brand'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'] == '(not set)') & (df_copy['device_category'] == 'mobile'), 'device_os'
    ] = 'Android'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()) & (df_copy['device_category'] == 'mobile'), 'device_os'
    ] = 'Android'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()) & (df_copy['device_category'] == 'tablet'), 'device_os'
    ] = 'Android'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()) & (df_copy['device_category'] == 'desktop'), 'device_os'
    ] = 'Windows'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()), 'device_os'
    ] = 'Android'

    basic_os = [
        'Android',
        'iOS',
        'Windows',
        'Macintosh',
        'Linux',
        'other_os'
    ]

    other_os_list = [i_os for i_os in df_copy['device_os'].values if i_os not in basic_os]
    df_copy['device_os'] = df_copy['device_os'].replace(other_os_list, 'other_os')

    df_copy.loc[(df_copy['device_brand'].isna()), 'device_brand'] = 'other_brand'

    basic_brands = [
        'Apple',
        'Samsung',
        'Xiaomi',
        'Huawei',
        'Realme',
        'OPPO',
        'Vivo'
        'other_brand',
    ]

    other_brands = [brand for brand in df_copy['device_brand'].values if brand not in basic_brands]
    df_copy['device_brand'] = df_copy['device_brand'].replace(list(set(other_brands)), 'other_brand')

    other_source = [source[0] for source in df_copy['utm_source'].value_counts(dropna=False).items() if
                    source[1] < 10 ** 3 or source[0] == np.nan]
    df_copy['utm_source'] = df_copy['utm_source'].replace(list(set(other_source)), 'other_source')

    other_medium = [medium[0] for medium in df_copy['utm_medium'].value_counts(dropna=False).items() if
                    medium[1] < 10 ** 3 or medium[0] == '(none)']
    df_copy['utm_medium'] = df_copy['utm_medium'].replace(list(set(other_medium)), 'other_medium')

    other_city = [city[0] for city in df_copy['geo_city'].value_counts(dropna=False).items() if
                  city[1] < 10 ** 3 or city[0] == '(not set)']
    df_copy['geo_city'] = df_copy['geo_city'].replace(list(set(other_city)), 'other_city')

    other_country = [country[0] for country in df_copy['geo_country'].value_counts(dropna=False).items() if
                     country[1] < 10 ** 4 or country[0] == '(not set)']
    df_copy['geo_country'] = df_copy['geo_country'].replace(other_country, 'other_country')

    return df_copy


def main():
    print('Prediction Pipeline')

    df = pd.read_csv('data/clean_df.csv').drop(columns=['session_id', 'client_id'], axis=1)

    X = df.drop('target_action', axis=1)
    y = df['target_action']

    columns_to_drop = [
        'visit_date',
        'visit_time',
        'device_screen_resolution',
    ]

    filter_features = Pipeline(
        steps=[
            ('filter', FunctionTransformer(delete_columns)),
            ('add_features', FunctionTransformer(add_new_features)),
            ('del_outliers', FunctionTransformer(delete_outliers)),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            # ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', FunctionTransformer(categorical_inputer)),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, make_column_selector(dtype_include='number')),
            ('categorical', categorical_transformer, make_column_selector(dtype_include='object')),
            ('column_dropper', 'drop', columns_to_drop),
        ]
    )

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC(),
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(
            steps=[
                ('filtering', filter_features),
                ('preprocessor', preprocessor),
                ('classifier', model),
            ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')

        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'model_pipe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
