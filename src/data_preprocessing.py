#write a code to handle categorical variables
#use one hot encoding

import numpy as np
import pandas as pd
from data_exploration.ipynb import df
from data_exploration.ipynb import categorical_cols

#Copy
df_copy= df.copy()

# One-hot encode categorical columns 
df_encoded = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)#.astype(int)
