import pandas as pd, numpy as np, sys
df=pd.read_csv(sys.argv[1],dtype='float32')
np.savez('criteo.npz',X=df[[f'f{i}' for i in range(12)]].to_numpy(),Z=df.treatment.to_numpy().astype(np.int8),Y=df.visit.to_numpy().astype(np.int8),C=df.conversion.to_numpy().astype(np.int8),D=df.exposure.to_numpy().astype(np.int8))
print(len(df)); print(df[['treatment','visit','conversion','exposure']].groupby('treatment').agg(['mean','count']))
print('exposure in control:',df.exposure[df.treatment==0].sum())
