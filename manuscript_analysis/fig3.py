import sys, numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, statsmodels.formula.api as smf
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
d['ibr'] = 100*d.inc/d.blink_rate; d['female'] = (d.sex==1).astype(int)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#555','axes.labelcolor':'#222','xtick.color':'#444','ytick.color':'#444'})
COL = {'HC':'#2a78d6','TH':'#eb6834','TO':'#1baf7a'}; M = {'HC':'o','TH':'s','TO':'^'}
LAB = {'HC':'Controls','TH':'AITD without TO','TO':'TO'}
fig = plt.figure(figsize=(7.2, 3.2), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.45)
ax = fig.add_subplot(gs[0])
s = d.dropna(subset=['mgl','ibr','age']).copy(); s['ibr10'] = s.ibr/10
s['group'] = pd.Categorical(s.group, ['HC','TO','TH'])
m = smf.mixedlm('mgl ~ ibr10 + C(group) + age + female', s, groups=s['pid']).fit()
b, (lo, hi), p = m.params['ibr10'], m.conf_int().loc['ibr10'], m.pvalues['ibr10']
rng = np.random.default_rng(1)
for g in ['HC','TH','TO']:
    t = s[s.group==g]
    ax.scatter(t.ibr + rng.uniform(-0.8,0.8,len(t)), t.mgl, s=16, marker=M[g], facecolor=COL[g], edgecolor='white', linewidth=0.5, alpha=0.9, label=f'{LAB[g]} (n={len(t)} eyes)', zorder=3)
# marginal line: model prediction at mean covariates, averaged over group mix
xs = np.linspace(0, 100, 50)
base = m.params['Intercept'] + m.params['age']*s.age.mean() + m.params['female']*s.female.mean() + \
       sum(m.params[f'C(group)[T.{g}]']*(s.group==g).mean() for g in ['TO','TH'])
ax.plot(xs, base + b*xs/10, color='#222', lw=1.6, zorder=4)
ax.set_xlabel('Incomplete blink ratio (%)'); ax.set_ylabel('Meibomian gland loss (%)')
ax.set_xlim(-4, 104); ax.set_ylim(0, 68); ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
ax.set_title(f'+{b:.2f}% per 10% IBR (95% CI {lo:.2f} to {hi:.2f}); P = {p:.3f}', fontsize=7, color='#222')
ax.legend(loc='upper left', fontsize=6.3, frameon=False, handletextpad=0.2, borderaxespad=0.2)
ax.text(-0.13, 1.04, 'A', transform=ax.transAxes, fontsize=11, fontweight='bold')
to = d[d.group=='TO'].copy(); to['retr'] = np.where(to.retraction==1, 'Retraction', 'No retraction')
for i, (v, ylab, L) in enumerate([('ibr','Incomplete blink ratio (%)','B'), ('mgl','Meibomian gland loss (%)','C')]):
    a = fig.add_subplot(gs[i+1])
    ns = []
    for j, k in enumerate(['No retraction','Retraction']):
        y = to.loc[to.retr==k, v].dropna()
        a.scatter(j + rng.uniform(-0.15,0.15,len(y)), y, s=12, marker='^', facecolor=COL['TO'], edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
        q1, med, q3 = y.quantile([.25,.5,.75])
        a.plot([j-0.28, j+0.28], [med, med], color='#222', lw=1.8, zorder=4)
        a.plot([j, j], [q1, q3], color='#222', lw=1, zorder=4)
        ns.append(len(y))
    tt = to.dropna(subset=[v,'age']).assign(r=lambda x: (x.retraction==1).astype(int))
    mm = smf.mixedlm(f'{v} ~ r + age', tt, groups=tt['pid']).fit()
    pv = mm.pvalues['r']
    a.set_title(f'P = {pv:.3f}', fontsize=7, color='#222')
    a.set_xticks([0,1]); a.set_xticklabels([f'No retraction\n(n={ns[0]})', f'Retraction\n(n={ns[1]})'], fontsize=6.8); a.set_xlim(-0.6,1.6)
    a.set_ylabel(ylab); a.grid(axis='y', color='#e6e6e6', lw=0.6); a.set_axisbelow(True)
    a.text(-0.32, 1.04, L, transform=a.transAxes, fontsize=11, fontweight='bold')
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17)
for ext in ['png','tiff']:
    fig.savefig(f'{out}/Figure3.{ext}', dpi=300, **({'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}))
print('beta', b, lo, hi, p)
