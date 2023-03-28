# -*- coding: utf-8 -*-
"""
@Time    : 3/15/2023 4:15 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Read the Excel file
df = pd.read_csv('Total_Detection_Delay.csv')
df = df.drop('Idx', axis=1)
df = df.drop('dirn', axis=1)
df = df.drop('run', axis=1)
cruise_dict = {}
for i in df.cruise_spd.unique():
    cruise_dict[i] = int((''.join(filter(str.isdigit, i))))/10  # hard code, convert string to integer
df['cruise_spd'] = df['cruise_spd'].map(cruise_dict)

# ds = df.isin([np.inf, -np.inf])
# print(ds)
# print(df.isnull().values.any())

# Fit the model using type III sum of squares
model = ols("total_delay_time ~ C(connection_type) * C(ext_mode) * C(offboard) * cruise_spd * GNSS_hor_std * GNSS_vert_std * GNSS_vel_hor_std * UR_UAS * UR_TRK * avail ", data=df).fit(method='pinv', cov_type='HC3')
# model = ols("total_delay_time ~ cruise_spd+avail + cruise_spd*avail", data=df).fit()
# model = ols("total_delay_time ~ C(cruise_spd, Sum) + C(avail, Sum)", data=df).fit()
# model = ols("total_delay_time ~ C(connection_type) + C(ext_mode) +C(offboard) + cruise_spd + GNSS_hor_std + GNSS_vert_std + GNSS_vel_hor_std + avail", data=df).fit(method='pinv', cov_type='HC3')

mod_sum = model.summary()

# summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
# print(summary_df)

# Perform the type III ANOVA
aov_table = sm.stats.anova_lm(model, typ=3)

# Print the results
print(aov_table)