import sys, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src).dropna(how='all').reset_index(drop=True)
df.columns = ['name','sex','age','thyroid','eye_inv','osdi','selenium','smoking','cas','side','retraction','hertel',
              'schirmer','nibut','pfw','oxford','blink_rate','inc','ibr','comp','mgl','meiboscore','dx']
for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',,', '.', regex=False).str.replace(',', '.', regex=False), errors='coerce')
# pair consecutive rows (right=1 then left=2) into patients; verify against name/age/group
assert (df.side.values[0::2] == 1).all() and (df.side.values[1::2] == 2).all(), 'rows not in R/L order'
df['pid'] = np.repeat(np.arange(1, len(df)//2 + 1), 2)
nm = df.name.astype(str).str.strip().str.lower()
pair_name_mismatch = int((nm.values[0::2] != nm.values[1::2]).sum())
for c in ['age','sex','thyroid','eye_inv','osdi','dx','selenium','smoking']:
    a, b = df[c].values[0::2], df[c].values[1::2]
    bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    if bad.any(): print('within-patient mismatch', c, 'pids', df.pid.values[0::2][bad].tolist())
print('pairs with differing name spelling:', pair_name_mismatch)
df['group'] = np.select([(df.thyroid==1)&(df.eye_inv==1), (df.thyroid==1)&(df.eye_inv==2), df.thyroid==2], ['TO','TH','HC'], default='')
df['ibr_calc'] = 100*df.inc/df.blink_rate
print('blink_rate identical in both eyes:', (df.blink_rate.values[0::2]==df.blink_rate.values[1::2]).mean())
print('inc+comp==blink_rate:', ((df.inc+df.comp)==df.blink_rate).mean(), '| |ibr - inc/blink*100|>1.5:', int((abs(df.ibr-df.ibr_calc)>1.5).sum()))
print('ibr out of range:', int(((df.ibr<0)|(df.ibr>100)).sum()), ' blink_rate==0:', int((df.blink_rate==0).sum()))
print('patients per group:', df.groupby('group').pid.nunique().to_dict())
df.drop(columns=['name']).to_csv(out, index=False)
print('saved de-identified', out)
