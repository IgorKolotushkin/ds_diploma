import datetime

import dill
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample


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

    basic_os = [
        'Android',
        'iOS',
        'Windows',
        'Macintosh',
        'Linux',
        'other_os'
    ]

    df_copy.loc[
        (df_copy['device_brand'].isin(['Huawei', 'Samsung'])) & (
                    df_copy['device_category'] == 'desktop'), 'device_category'
    ] = 'mobile'

    df_copy.loc[
        (df_copy['device_brand'].isin(['Xiaomi', 'Huawei', 'Samsung'])) & (df_copy['device_os'].isna()) & (
            df_copy['device_category'].isin(device_list)), 'device_os'
    ] = 'Android'

    df_copy.loc[
        (df_copy['device_brand'] == 'Apple') & (df_copy['device_category'].isin(device_list)) & (
            df_copy['device_os'].isna()), 'device_os'
    ] = 'iOS'
    df_copy.loc[
        (df_copy['device_brand'] == 'Apple') & (df_copy['device_category'] == 'desktop') & (
            df_copy['device_os'].isna()), 'device_os'
    ] = 'Macintosh'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == 'Windows'), 'device_brand'
    ] = 'other_brand'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'].isna()) & (
                    df_copy['device_category'] == 'desktop'), 'device_os'
    ] = 'Windows'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'] == 'Macintosh'), 'device_brand'
    ] = 'Apple'
    df_copy.loc[
        (df_copy['device_brand'].isna()) & (df_copy['device_os'].isin(['(not set)', 'Chrome OS'])) & (
                    df_copy['device_category'] == 'desktop'), 'device_brand'
    ] = 'other_brand'

    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'] == '(not set)') & (
                    df_copy['device_category'] == 'mobile'), 'device_os'
    ] = 'Android'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()) & (
            df_copy['device_category'].isin(device_list)), 'device_os'
    ] = 'Android'

    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()) & (
                    df_copy['device_category'] == 'desktop'), 'device_os'
    ] = 'Windows'
    df_copy.loc[
        (df_copy['device_brand'].notna()) & (df_copy['device_os'].isna()), 'device_os'
    ] = 'Android'

    other_os_list = [os for os in df_copy['device_os'].values if os not in basic_os]
    df_copy['device_os'] = df_copy['device_os'].replace(other_os_list, 'other_os')

    df_copy.loc[(df_copy['device_brand'].isna()), 'device_brand'] = 'other_brand'

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


def balance_dataframe(df):
    target_0 = df[df.target_action == 0]
    target_1 = df[df.target_action == 1]

    target_0_downsampled = resample(
        target_0,
        replace=False,
        n_samples=len(target_1),
        random_state=27,
    )

    return pd.concat([target_0_downsampled, target_1])


def main():
    print('Prediction Pipeline')

    df = pd.read_csv('data/df_with_target.csv').drop(columns=['session_id', 'client_id'], axis=1)

    df = balance_dataframe(df)

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
        LogisticRegression(
            C=1,
            max_iter=150,
            random_state=42,
            solver='newton-cg',
        ),
        RandomForestClassifier(
            n_estimators=150,
            min_samples_split=3,
            random_state=42,
        ),
        SVC(
            random_state=42,
        ),
        GradientBoostingClassifier(
            n_estimators=150,
            random_state=42,
        )
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(
            steps=[
                ('filtering', filter_features),
                ('preprocessor', preprocessor),
                ('classifier', model),
            ]
        )

        score = cross_val_score(pipe, X, y, cv=3, scoring='roc_auc')

        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        best_pipe = pipe
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(X, y)

    model_data = {
        'model': best_pipe,
        'metadata': {
            'name': 'Client Prediction Pipeline',
            'author': 'Igor Kolotushkin',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'roc_auc': best_score,
        }
    }
    with open('model_pipe.pkl', 'wb') as file:
        dill.dump(model_data, file, recurse=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
