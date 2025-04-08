import pandas as pd
import statsmodels.api as sm
import numpy as np


train_path = ""

train_df = pd.read_csv(train_path, encoding='gbk')

features = train_df.columns.drop(['id', 'label'])
significant_features_single = []

single_logit_results = {}
for feature in features:
    try:
        model = sm.Logit(train_df['label'], sm.add_constant(train_df[feature])).fit(disp=0)
        params = model.params
        conf = model.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        conf_exp = np.exp(conf)
        single_logit_results[feature] = conf_exp
        if model.pvalues[feature] < 0.1:
            significant_features_single.append(feature)
    except Exception as e:
        print(f"Error processing feature {feature}: {e}")

print(significant_features_single)
for feature, result in single_logit_results.items():
    print(f"{feature}:\n{result}\n")


X = sm.add_constant(train_df[significant_features_single]) 
y = train_df['label']
model = sm.Logit(y, X).fit(disp=0)
print(model.summary())

params = model.params
conf = model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
conf_exp = np.exp(conf)
print(conf_exp)
