"""Step 7: eldercare-onset event study (Eq. 5), exit from eldercare and the childcare comparison event.
Onset = first observed transition from 0 to > 0 eldercare hours. Persons already caring in their first observed
wave are excluded (no observable onset); never-starters are controls and later starters contribute pre-onset
observations. Exit (descriptive) = person-years from the first observed exit onward, compared with the same
person's caring person-years; a person enters the exit sample at their first observed caring wave, so baseline
carers are retained. Same covariates, fixed effects and clustering as Eq. (1). Every record carries the number
of clusters of its own estimation sample (G) and a t(G-1) 95% interval (lo, hi), which the figures read.
Outputs: output/main/table3_eventstudy.csv/.md;
         output/supplement/es_dynamic.csv, es_dose.csv, es_pretrend.csv, tableS5_es_robustness.csv/.md, tableS5b_es_stacked.csv"""
import os, numpy as np, pandas as pd
from common import *

P = load_panel(); S = analytic_sample(P); G = groups(S)
A = pd.concat([d.assign(grp=g) for g, d in G.items()])          # the three estimation samples of Table 1 (persons observed twice within the group)
GROUPS = ['Shrinking SMCs', 'Non-shrinking SMCs', 'Large cities']

def build(sub, var='Elder_h', exclude_baseline_carers=True):
    """Event-time variables for the onset of `var` (> 0)."""
    s = sub.sort_values(['PID', 'Year']).copy(); s['care'] = (s[var] > 0).astype(int)
    first_wave = s.groupby('PID').Year.transform('min'); base = s[s.Year == first_wave].set_index('PID').care
    s['base_care'] = s.PID.map(base)
    if exclude_baseline_carers: s = s[s.base_care == 0].copy()
    s['prev'] = s.groupby('PID').care.shift(1); s['start'] = ((s.care == 1) & (s.prev == 0)).astype(int)
    first = s[s.start == 1].groupby('PID').Year.min(); s['t0'] = s.PID.map(first); s['rel'] = s.Year - s.t0
    s['post'] = (s.rel >= 0).astype(int).fillna(0)
    s['em3'] = (s.rel <= -3).astype(int); s['em2'] = (s.rel == -2).astype(int); s['e0'] = (s.rel == 0).astype(int); s['e1'] = (s.rel == 1).astype(int); s['e2p'] = (s.rel >= 2).astype(int)
    h0 = s[s.rel == 0].set_index('PID')[var]; s['h0'] = s.PID.map(h0)
    s['post_light'] = ((s.post == 1) & (s.h0 < 1)).astype(int); s['post_heavy'] = ((s.post == 1) & (s.h0 >= 1)).astype(int)
    return s

def fit_es(s, rhs, extra='', cov=COV):
    return pf.feols(f"LS0 ~ {rhs}{extra} + C(Age){cov} | PID + scale_year", data=s, vcov={'CRV1': 'muni'})

def coef(f, k): return f.coef()[k], f.se()[k], f.pvalue()[k]
def onsets(s): return int(s[s.t0.notna()].PID.nunique())
def rec(f, k, **kw):
    """One estimate with SE, p (pyfixest, t(G-1)), the cluster count of the estimation sample and a t(G-1) 95% CI."""
    e, se, p = coef(f, k); Gf = n_clusters(f); cv = crit(Gf)
    return dict(est=e, se=se, p=p, stars=stars(p), clusters=Gf, crit=cv, lo=e - cv * se, hi=e + cv * se, N=f._N, **kw)

static, dyn, dose, pretrend = [], [], [], []
from scipy.stats import chi2
for g in GROUPS:
    sub = A[A.grp == g]
    s = build(sub); f = fit_es(s, 'post')
    static.append(rec(f, 'post', group=g, event='Eldercare onset', onsets=onsets(s)))
    fe = fit_es(s, 'em3 + em2 + e0 + e1 + e2p')
    bb = fe.coef()[['em3', 'em2']].values; VV = pd.DataFrame(fe._vcov, index=fe.coef().index, columns=fe.coef().index).loc[['em3', 'em2'], ['em3', 'em2']].values
    W = float(bb @ np.linalg.solve(VV, bb)); pretrend.append(dict(group=g, wald_chi2=W, df=2, p=1 - chi2.cdf(W, 2)))   # joint test of the two leads
    for k, lab, n in [('em3', '<=-3', int((s.rel <= -3).sum())), ('em2', '-2', int((s.rel == -2).sum())), ('e0', '0', int((s.rel == 0).sum())), ('e1', '+1', int((s.rel == 1).sum())), ('e2p', '>=+2', int((s.rel >= 2).sum()))]:
        dyn.append(rec(fe, k, group=g, rel=lab, n=n))
    dyn.append(dict(group=g, rel='-1 (ref)', est=0.0, se=0.0, p=np.nan, stars='', clusters=n_clusters(fe), crit=crit(n_clusters(fe)), lo=0.0, hi=0.0, N=fe._N, n=int((s.rel == -1).sum())))
    fd = fit_es(s, 'post_light + post_heavy')
    for k, lab, cond in [('post_light', 'Onset at 0.5 h/day', s.h0 < 1), ('post_heavy', 'Onset at >= 1 h/day', s.h0 >= 1)]:
        dose.append(rec(fd, k, group=g, intensity=lab, onsets=int(s[(s.rel == 0) & cond].PID.nunique())))
    # exit from eldercare (descriptive): person-years from the first observed exit onward, compared with the same person's
    # caring person-years. A person enters at their first observed caring wave (baseline carers are retained, and a person
    # who stops and later resumes keeps their first spell and exit); never-carers remain as controls.
    s2 = sub.sort_values(['PID', 'Year']).copy(); s2['care'] = (s2.Elder_h > 0).astype(int)
    first_care = s2[s2.care == 1].groupby('PID').Year.min(); fc_year = s2.PID.map(first_care)
    s2 = s2[fc_year.isna() | (s2.Year >= fc_year)].copy()
    s2['prev'] = s2.groupby('PID').care.shift(1); s2['stop'] = ((s2.care == 0) & (s2.prev == 1)).astype(int)
    t1 = s2[s2.stop == 1].groupby('PID').Year.min(); s2['poststop'] = ((s2.Year - s2.PID.map(t1)) >= 0).astype(int).fillna(0)
    fs = fit_es(s2, 'poststop'); static.append(rec(fs, 'poststop', group=g, event='Eldercare exit', onsets=int(s2[s2.poststop == 1].PID.nunique())))
    # comparison event: childcare onset through the same mechanism
    sc = build(sub, var='Child_h'); fc = fit_es(sc, 'post'); static.append(rec(fc, 'post', group=g, event='Childcare onset (comparison event)', onsets=onsets(sc)))
# pooled contrasts
s = build(A); s['shrd'] = (s.grp == 'Shrinking SMCs').astype(int); s['grwd'] = (s.grp == 'Non-shrinking SMCs').astype(int)
f = fit_es(s, 'post + post:shrd + post:grwd')
for k, lab in [('post', 'Large cities (reference)'), ('post:shrd', 'Difference: shrinking SMCs - large cities'), ('post:grwd', 'Difference: non-shrinking SMCs - large cities')]:
    static.append(rec(f, k, group='pooled', event=lab, onsets=onsets(s)))
s = build(A[A.scale == 'SMC']); s['shrd'] = (s.grp == 'Shrinking SMCs').astype(int); f = fit_es(s, 'post + post:shrd')
static.append(rec(f, 'post:shrd', group='pooled', event='Difference: shrinking - non-shrinking (within SMCs)', onsets=onsets(s)))
# robustness for shrinking SMCs
sub = A[A.grp == 'Shrinking SMCs']; rob = []
s0 = build(sub)
s0 = s0.sort_values(['PID', 'Year']); s0['prev_year'] = s0.groupby('PID').Year.shift(1)
gap_pids = set(s0[(s0.rel == 0) & (s0.Year - s0.prev_year != 1)].PID)                      # onset year not pinned down
specs = [('Main', s0, 'post', '', COV), ('Controlling the other six time-use domains', s0, 'post', ' + Sleep_h + Work_h + Child_h + Commute_h + Educ_h + Free_h', COV),
         ('Onsets observed in consecutive years only', s0[~s0.PID.isin(gap_pids)], 'post', '', COV),
         ('Post restricted to person-years with eldercare > 0', s0[~((s0.post == 1) & (s0.Elder_h == 0))], 'post', '', COV),
         ('Stayers only', build(sub[sub.stayer]), 'post', '', COV), ('No time-varying covariates', s0, 'post', '', ''), ('2022–2024 waves only', build(sub[sub.Year >= 2022]), 'post', '', COV)]
s_ = sub.copy(); s_['Elder_h'] = np.where(s_.Elder_h >= 1, s_.Elder_h, 0); specs.append(('Onset defined at ≥ 1 h/day', build(s_), 'post', '', COV))
for lab, s_, k, extra, cov in specs:
    f = fit_es(s_, k, extra, cov); rob.append(rec(f, k, spec=lab, onsets=onsets(s_)))
s_ = s0.copy(); s_['post_caring'] = ((s_.post == 1) & (s_.Elder_h > 0)).astype(int); s_['post_after_exit'] = ((s_.post == 1) & (s_.Elder_h == 0)).astype(int)
f = fit_es(s_, 'post_caring + post_after_exit')
for k, lab in [('post_caring', 'Post split: person-years while caring'), ('post_after_exit', 'Post split: person-years after exit from care')]:
    rob.append(rec(f, k, spec=lab, onsets=onsets(s_)))
# stacked DID with never-treated clean controls, cohort-specific person and year effects, window -3..+2
def stacked(sub, lo=-3, hi=2):
    s = build(sub); nev = s[s.t0.isna()]; blocks = []
    for c in sorted(s.t0.dropna().unique()):
        tr = s[(s.t0 == c) & ((s.Year - c).between(lo, hi))].copy()
        ct = nev[(nev.Year - c).between(lo, hi)].copy(); ct['rel'] = ct.Year - c; ct['post'] = 0
        for k in ['em3', 'em2', 'e0', 'e1', 'e2p']: ct[k] = 0
        blk = pd.concat([tr, ct]); blk['cohort'] = int(c); blk['pid_c'] = blk.PID.astype(str) + '_' + str(int(c)); blk['yr_c'] = blk.Year.astype(str) + '_' + str(int(c))
        n = blk.groupby('pid_c').Year.transform('nunique'); blocks.append(blk[n >= 2])
    st = pd.concat(blocks)
    f = pf.feols(f"LS0 ~ post + C(Age){COV} | pid_c + yr_c", data=st, vcov={'CRV1': 'muni'})
    fd = pf.feols(f"LS0 ~ em3 + em2 + e0 + e1 + e2p + C(Age){COV} | pid_c + yr_c", data=st, vcov={'CRV1': 'muni'})
    return f, fd, st
stk = []
for g in GROUPS:
    f, fd, st = stacked(A[A.grp == g])
    r_ = rec(f, 'post', group=g, onsets=int(st[st.t0.notna()].PID.nunique()), N_unique_person_years=int(st[['PID', 'Year']].drop_duplicates().shape[0]), cohorts=int(st.cohort.nunique())); r_['N_stacked'] = r_.pop('N')
    for k, cond in [('em3', st.rel <= -3), ('em2', st.rel == -2), ('e0', st.rel == 0), ('e1', st.rel == 1), ('e2p', st.rel >= 2)]:
        r_[f'dyn_{k}'] = fd.coef()[k]; r_[f'dyn_{k}_se'] = fd.se()[k]; r_[f'dyn_{k}_p'] = fd.pvalue()[k]; r_[f'dyn_{k}_n_treated'] = int(((cond) & st.t0.notna()).sum())
    stk.append(r_)
STK = pd.DataFrame(stk); STK.to_csv(os.path.join(OUT_S, 'tableS5b_es_stacked.csv'), index=False); pd.DataFrame(pretrend).to_csv(os.path.join(OUT_S, 'es_pretrend.csv'), index=False)
ST = pd.DataFrame(static); DY = pd.DataFrame(dyn); DO = pd.DataFrame(dose); RO = pd.DataFrame(rob)
ST.to_csv(os.path.join(OUT_T, 'table3_eventstudy.csv'), index=False); DY.to_csv(os.path.join(OUT_S, 'es_dynamic.csv'), index=False); DO.to_csv(os.path.join(OUT_S, 'es_dose.csv'), index=False); RO.to_csv(os.path.join(OUT_S, 'tableS5_es_robustness.csv'), index=False)
# markdown Table 3: post-onset average only; intensity and comparison events are Table S5(a)
F = lambda r: fmt(r.est, r.se, r.p, G=int(r.clusters))          # t(G-1) with the cluster count of each model's own estimation sample
rows = []
for g in GROUPS:
    a = ST[(ST.group == g) & (ST.event == 'Eldercare onset')].iloc[0]; rows.append([g, F(a), str(int(a.onsets)), f'{int(a.N):,}'])
for ev in ['Difference: shrinking SMCs - large cities', 'Difference: non-shrinking SMCs - large cities', 'Difference: shrinking - non-shrinking (within SMCs)']:
    r = ST[(ST.group == 'pooled') & (ST.event == ev)].iloc[0]; rows.append([ev.replace(' - ', ' − '), F(r), str(int(r.onsets)), f'{int(r.N):,}'])
T4 = pd.DataFrame(rows, columns=['Group (adults)', 'Post-onset average', 'Onsets', 'Person-years'])
to_md(T4, os.path.join(OUT_T, 'table3_eventstudy.md'))
with open(os.path.join(OUT_T, 'table3_eventstudy.md'), 'a', encoding='utf-8') as fh:
    fh.write('\nNote: 95% t(G−1) intervals with G the number of municipality clusters of each model. Intensity at onset, the comparison events (exit from eldercare, onset of childcare) and the robustness checks are in Table S5.\n')
print(T4.to_string(index=False)); print(DY.round(3).to_string(index=False)); print(pd.DataFrame(pretrend).round(3).to_string(index=False))
# Table S5: (a) intensity and comparison events for the three groups, (b) alternative specifications for shrinking SMCs, (c) stacked DID
rows7a = []
for g in GROUPS:
    ex = ST[(ST.group == g) & (ST.event == 'Eldercare exit')].iloc[0]; ch = ST[(ST.group == g) & (ST.event == 'Childcare onset (comparison event)')].iloc[0]
    dl = DO[(DO.group == g) & (DO.intensity == 'Onset at 0.5 h/day')].iloc[0]; dh = DO[(DO.group == g) & (DO.intensity == 'Onset at >= 1 h/day')].iloc[0]
    rows7a.append([g, F(dl), F(dh), F(ex), F(ch), f'{int(dl.onsets)} / {int(dh.onsets)}', f'{int(ex.onsets)} ({int(ex.N):,})', f'{int(ch.onsets)} ({int(ch.N):,})'])
T7a = pd.DataFrame(rows7a, columns=['Group (adults)', 'Onset at 0.5 h/day', 'Onset at ≥ 1 h/day', 'Eldercare exit (since first observed exit, vs. caring years)', 'Childcare onset (comparison event)', 'Onsets (0.5 / ≥ 1 h)', 'Exits (person-years)', 'Childcare onsets (person-years)'])
rows7 = [[r.spec, fmt(r.est, r.se, r.p, G=int(r.clusters)), str(int(r.onsets)), f'{int(r.N):,}'] for _, r in RO.iterrows()]
T7b = pd.DataFrame(rows7, columns=['Specification (shrinking SMCs)', 'Post-onset average', 'Onsets', 'Person-years'])
rows7c = [[r.group, fmt(r.est, r.se, r.p, G=int(r.clusters)), str(int(r.onsets)), f'{int(r.N_unique_person_years):,} ({int(r.N_stacked):,} stacked)'] for _, r in STK.iterrows()]
T7c = pd.DataFrame(rows7c, columns=['Group (adults)', 'Post-onset average', 'Onsets', 'Person-years (stacked)'])
with open(os.path.join(OUT_S, 'tableS5_es_robustness.md'), 'w', encoding='utf-8') as fh:
    fh.write('(a) Intensity at onset and comparison events, three groups\n\n' + md_str(T7a)
             + '\n(b) Alternative specifications, shrinking SMCs\n\n' + md_str(T7b)
             + '\n(c) Stacked difference-in-differences with never-treated controls only (cohort-specific person and year effects; window −3 to +2 years)\n\n' + md_str(T7c)
             + "\nNotes: Adults aged 15–64. Entries are changes in life satisfaction (0–10 scale) from the static event-study model (Eq. (5) of the manuscript) with person and city-size × year fixed effects, 5-year age bands and time-varying covariates; 95% t(G−1) intervals with municipality-clustered standard errors. (a) Intensity is the eldercare reported in the onset wave. Exit: a person enters at their first observed caring wave; person-years from the first observed exit onward are compared with the same person's caring person-years; never-carers are controls. Childcare onset is defined in the same way as eldercare onset (first observed transition from no childcare to any childcare). With five waves no cohort spans the full stacked window and the stack is unweighted, so (c) is a sensitivity check rather than a correction for staggered timing; dynamic coefficients of the stacked models are in tableS5b_es_stacked.csv.\n")
print(T7a.to_string(index=False)); print(T7b.to_string(index=False)); print(T7c.to_string(index=False))
