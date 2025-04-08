import pandas as pd
from scipy.stats import pearsonr


icc0 = pd.read_csv("")
icc1 = pd.read_csv("")
icc2 = pd.read_csv("")


group_inner = pd.merge(icc0, icc1, on='id_name', suffixes=('_0', '_1'))

group_between = pd.merge(icc0, icc2, on='id_name', suffixes=('_0', '_2'))


def calculate_correlation(df, suffix1, suffix2):
    correlations = {}
    for column in df.columns:
        if column.endswith(suffix1):
            feature1 = column
            feature2 = column.replace(suffix1, suffix2)
            if feature2 in df.columns:
                correlation, _ = pearsonr(df[feature1], df[feature2])
                correlations[column.replace(suffix1, '')] = correlation
    return correlations


inner_correlations = calculate_correlation(group_inner, '_0', '_1')
inner_df = pd.DataFrame(list(inner_correlations.items()), columns=['Feature', 'Correlation'])


between_correlations = calculate_correlation(group_between, '_0', '_2')
between_df = pd.DataFrame(list(between_correlations.items()), columns=['Feature', 'Correlation'])

inner_df.to_csv("Inner_Correlations.csv", index=False)
between_df.to_csv("Between_Correlations.csv", index=False)
