import pandas as pd
from pycombat import Combat


batch_col = 'Machine'

combat = Combat()

features = full_df.columns.difference(['id_name', 'Machine', 'label'])
full_df[features] = combat.fit_transform(Y=full_df[features].values, b=full_df[batch_col].values)

