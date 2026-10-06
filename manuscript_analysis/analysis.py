import sys, warnings, json, pandas as pd, numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr, chi2_contingency, fisher_exact
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
d = pd.read_csv(sys.argv[1]); G = ['TO','TH','HC']
d['ibr'] = 100*d.inc/d.blink_rate          # recomputed from raw counts
d['female'] = (d.sex==1).astype(int)       # coding to be confirmed by author
d['group'] = pd.Categorical(d.group, G)
eye_vars = ['hertel','pfw','schirmer','nibut','mgl','inc','ibr']
pt = d.groupby('pid').agg(group=('group','first'), age=('age','first'), female=('female','first'), osdi=('osdi','mean'),
        blink_rate=('blink_rate','first'), smoking=('smoking','first'), selenium=('selenium','first'), dx=('dx','first'),
        cas=('cas','max'), retraction_any=('retraction', lambda s: int((s==1).any())),
        **{v:(v,'mean') for v in eye_vars}).reset_index()
pt['group'] = pd.Categorical(pt.group, G)
L = []
def P(*a): L.append(' '.join(str(x) for x in a))
fmt = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'

P('## Patients per group'); P(pt.group.value_counts().reindex(G).to_dict())
P('## Missing eyes per group (mgl, schirmer, nibut, age, osdi)')
for v in ['mgl','schirmer','nibut','age','osdi']: P(v, d[d[v].isna()].groupby('group').size().reindex(G, fill_value=0).to_dict())

P('\n## Table 1 categorical (patient level)')
def cat(v, lab):
    t = pd.crosstab(pt[v], pt.group).reindex(columns=G, fill_value=0)
    p = chi2_contingency(t)[1] if t.shape[0] > 1 else float('nan')
    P(lab, {int(k): t.loc[k].tolist() for k in t.index}, 'col%:', (100*t/t.sum()).round(1).loc[t.index[0]].tolist(), 'p=', fmt(p))
cat('female','female(code1)'); cat('smoking','smoking 1=no 2=yes'); cat('selenium','selenium 1=yes 2=no'); cat('dx','dx 0=none 1=Graves 2=Hashimoto'); cat('cas','CAS max'); cat('retraction_any','retraction any eye')

P('\n## Continuous, patient level (mean of eyes): mean±SD | median[IQR] | KW p | pairwise MWU Bonferroni p (TO-TH, TO-HC, TH-HC)')
res = {}
for v in ['age','osdi','hertel','pfw','schirmer','nibut','mgl','blink_rate','inc','ibr']:
    g = [pt.loc[pt.group==k, v].dropna() for k in G]
    H, p = kruskal(*g)
    pw = [min(1, 3*mannwhitneyu(g[i], g[j]).pvalue) for i, j in [(0,1),(0,2),(1,2)]]
    s = ' | '.join(f'{k} {x.mean():.2f}±{x.std():.2f} med {x.median():.1f}[{x.quantile(.25):.1f}-{x.quantile(.75):.1f}] n={len(x)}' for k, x in zip(G, g))
    P(f'{v}: {s} | H={H:.2f} p={fmt(p)} | pw {[fmt(q) for q in pw]}')
    res[v] = dict(p=p, pw=pw)

P('\n## Mixed models (eye level, random intercept per patient), adjusted for age + female. Ref = HC. coef [95%CI] p')
def mm(formula, data, terms):
    m = smf.mixedlm(formula, data, groups=data['pid']).fit(reml=True)
    ci = m.conf_int()
    return {t: (m.params[t], ci.loc[t,0], ci.loc[t,1], m.pvalues[t]) for t in terms}
dd = d.copy(); dd['group'] = dd.group.cat.reorder_categories(['HC','TO','TH'])
for v in ['osdi','nibut','schirmer','mgl','hertel','pfw','blink_rate','inc','ibr']:
    sub = dd.dropna(subset=[v,'age'])
    r = mm(f'{v} ~ C(group) + age + female', sub, ['C(group)[T.TO]','C(group)[T.TH]','age'])
    # TO vs TH
    sub2 = sub.copy(); sub2['group'] = sub2.group.cat.reorder_categories(['TH','TO','HC'])
    r2 = mm(f'{v} ~ C(group) + age + female', sub2, ['C(group)[T.TO]'])
    P(v, ' ; '.join(f'{k.replace("C(group)[T.","").replace("]","")}: {a:.2f} [{lo:.2f}, {hi:.2f}] p={fmt(p)}' for k,(a,lo,hi,p) in r.items()),
      f'; TO-vs-TH: {r2["C(group)[T.TO]"][0]:.2f} [{r2["C(group)[T.TO]"][1]:.2f}, {r2["C(group)[T.TO]"][2]:.2f}] p={fmt(r2["C(group)[T.TO]"][3])}')

P('\n## Inter-eye ICC (null mixed model)')
for v in ['ibr','inc','mgl','nibut','schirmer']:
    sub = d.dropna(subset=[v]); m = smf.mixedlm(f'{v} ~ 1', sub, groups=sub['pid']).fit()
    P(v, 'ICC=', round(float(m.cov_re.iloc[0,0]/(m.cov_re.iloc[0,0]+m.scale)), 2))

P('\n## IBR vs MGL')
r, p = spearmanr(d.ibr, d.mgl, nan_policy='omit'); P('eye-level pooled Spearman (naive)', round(r,3), fmt(p), 'n eyes', d[['ibr','mgl']].dropna().shape[0])
r, p = spearmanr(pt.ibr, pt.mgl, nan_policy='omit'); P('patient-level pooled Spearman', round(r,3), fmt(p), 'n pts', pt[['ibr','mgl']].dropna().shape[0])
for k in G:
    s = pt[pt.group==k]; r, p = spearmanr(s.ibr, s.mgl, nan_policy='omit'); P(f' patient-level {k}', round(r,3), fmt(p), 'n', s[['ibr','mgl']].dropna().shape[0])
    s = d[d.group==k]; r, p = spearmanr(s.ibr, s.mgl, nan_policy='omit'); P(f' eye-level {k} (naive)', round(r,3), fmt(p))
sub = dd.dropna(subset=['mgl','ibr','age']); sub = sub.assign(ibr10=sub.ibr/10, inc_=sub.inc)
for f in ['mgl ~ ibr10', 'mgl ~ ibr10 + age + female', 'mgl ~ ibr10 + C(group) + age + female']:
    r = mm(f, sub, ['ibr10']); a,lo,hi,p = r['ibr10']; P(f'MM {f}: per 10% IBR {a:.2f} [{lo:.2f}, {hi:.2f}] p={fmt(p)}')
r = mm('mgl ~ inc_ + C(group) + age + female', sub, ['inc_']); a,lo,hi,p = r['inc_']; P(f'MM mgl ~ incomplete count + group + age + female: {a:.2f} [{lo:.2f}, {hi:.2f}] p={fmt(p)}')
for k in G:
    s = sub[sub.group==k]; r = mm('mgl ~ ibr10 + age', s, ['ibr10']); a,lo,hi,p = r['ibr10']; P(f' MM within {k}: per 10% IBR {a:.2f} [{lo:.2f}, {hi:.2f}] p={fmt(p)} n_eyes={len(s)}')
r = mm('mgl ~ ibr10 * C(group) + age + female', sub, []) ; 
m = smf.mixedlm('mgl ~ ibr10 * C(group) + age + female', sub, groups=sub['pid']).fit()
P(' interaction p:', {k: fmt(v) for k, v in m.pvalues.items() if ':' in k})

P('\n## Other blink correlations (patient level, Spearman, pooled & by group)')
for x, y in [('blink_rate','osdi'),('blink_rate','nibut'),('blink_rate','schirmer'),('ibr','osdi'),('ibr','nibut'),('ibr','hertel'),('ibr','pfw'),('inc','hertel'),('inc','pfw')]:
    out = []
    for lab, s in [('all', pt)] + [(k, pt[pt.group==k]) for k in G]:
        r, p = spearmanr(s[x], s[y], nan_policy='omit'); out.append(f'{lab} {r:.2f} (p={fmt(p)})')
    P(f'{x} vs {y}: ' + ' | '.join(out))

P('\n## TO: eyelid retraction (eye level mixed model, + age)')
to = d[d.group=='TO'].copy(); to['retr'] = (to.retraction==1).astype(int)
P('eyes with retraction', int(to.retr.sum()), 'patients with any', int(to.groupby('pid').retr.max().sum()), 'bilateral', int((to.groupby('pid').retr.sum()==2).sum()))
for v in ['ibr','inc','mgl']:
    s = to.dropna(subset=[v,'age']); r = mm(f'{v} ~ retr + age', s, ['retr']); a,lo,hi,p = r['retr']
    P(f'{v}: retraction {s[s.retr==1][v].mean():.2f}±{s[s.retr==1][v].std():.2f} vs none {s[s.retr==0][v].mean():.2f}±{s[s.retr==0][v].std():.2f}; adj diff {a:.2f} [{lo:.2f}, {hi:.2f}] p={fmt(p)}')

P('\n## Eye-level categorical (for reference; chi2 ignores pairing)')
for v in ['oxford','meiboscore']:
    t = pd.crosstab(d[v], d.group).reindex(columns=G, fill_value=0); P(v, {int(k): t.loc[k].tolist() for k in t.index}, 'p=', fmt(chi2_contingency(t)[1]))
t = pt.assign(ox=d.groupby('pid').oxford.max().values).pipe(lambda x: pd.crosstab(x.ox, x.group)).reindex(columns=G, fill_value=0)
P('oxford>=1 in any eye (patients)', {int(k): t.loc[k].tolist() for k in t.index}, 'p=', fmt(chi2_contingency(t)[1]))
open(sys.argv[2], 'w').write('\n'.join(L)); pt.to_csv(sys.argv[3], index=False); print('\n'.join(L))
