"""Step 6: sensitivity of the shrinking-SMC headline estimates (Table S3).
Three quantities per specification: the eldercare value, the sleep value (deviation-coded) and the eldercare -> sleep
reallocation, for S0 (main specification) and S1-S23 in five panels: (a) covariates, (b) total-hours window,
(c) sample and instrument period, (d) measurement, (e) context and inference.
Output: output/supplement/tableS3_sensitivity.csv/.md"""
import os, numpy as np, pandas as pd
from common import *

P = load_panel(); S = analytic_sample(P); G = groups(S)
base_all = P[P.Year.between(*YEARS) & (P.scale == 'SMC') & (P.age_grp == '15-64')]
shr = lambda d, thr=SHRINK_THRESHOLD: d[d.cagr_17_23 <= thr]
win = lambda d, w=TOTAL_WINDOW: d[d.total_h.between(*w)] if w else d
B = shr(win(base_all))
rows = []
PANELS = {'a': '(a) Covariates', 'b': '(b) Total-hours window', 'c': '(c) Sample and instrument period', 'd': '(d) Measurement', 'e': '(e) Context and inference'}
def record(label, f, b, V, d, doms=DOMS, persons=None, munis=None, panel='a'):
    nm = list(b.index); out = dict(spec=label, panel=PANELS[panel], N=f._N, persons=persons or d.PID.nunique(), munis=munis or d.muni.nunique())
    for key, c in [('elder', dev(nm, 'Elder', doms=doms)), ('sleep', dev(nm, 'Sleep', doms=doms)), ('elder_to_sleep', sub_c(nm, 'Elder', 'Sleep'))]:
        e, se, p = est(c, b, V); out.update({f'{key}_est': e, f'{key}_se': se, f'{key}_p': p, f'{key}_fmt': fmt(e, se, p, G=V.attrs.get('G'))})
    rows.append(out); print(f"{label:<58} elder {out['elder_fmt']:<28} e->s {out['elder_to_sleep_fmt']:<28} N={f._N:,}")
def run(label, sub, doms=DOMS, cov=COV, tot='total_h', vcov=None, fe='PID + scale_year', panel='a'):
    sub = panelize(sub); rhs = ' + '.join(f'{x}_h' for x in doms if x != 'Free')
    f, b, V = fit(sub, cov=cov, tot=tot, rhs=rhs, vcov=vcov, fe=fe); record(label, f, b, V, sub, doms, panel=panel)

# (a) covariates
run('S0 Main specification', B)
run('S1 No time-varying covariates', B, cov='')
run('S2 + self-rated health', B.dropna(subset=['srh']), cov=COV + ' + C(srh)')
run('S3 + self-rated health and personal income (2022–2024)', B[B.Year >= 2022].dropna(subset=['pinc', 'srh']), cov=COV + ' + C(srh) + C(pinc)')
# (b) total-hours window: a curve rather than a single alternative
run('S4 Total hours 8–24', shr(win(base_all, (8, 24))), panel='b')
run('S5 Total hours 6–30', shr(win(base_all, (6, 30))), panel='b')
run('S6 Total hours 4–36', shr(win(base_all, (4, 36))), panel='b')
run('S7 Total hours 1–48', shr(win(base_all, (1, 48))), panel='b')
run('S8 No total-hours window', shr(base_all), panel='b')
run('S9 Lower bound only (≥ 10 h)', shr(base_all[base_all.total_h >= 10]), panel='b')
run('S10 Upper bound only (≤ 26 h)', shr(base_all[base_all.total_h <= 26]), panel='b')
# (c) sample and instrument period
run('S11 Stayers only (one municipality in every observed wave)', B[B.stayer], panel='c')
run('S12 2022–2024 waves only', B[B.Year >= 2022], panel='c')
run('S13 2020–2021 waves only', B[B.Year <= 2021], panel='c')
d = B.copy(); d['pid_period'] = d.PID.astype(str) + np.where(d.Year >= 2022, '_b', '_a')          # person x instrument-period effects
n = d.groupby('pid_period').Year.transform('nunique'); d = d[n >= 2]
f, b, V = fit(d, fe='pid_period + scale_year'); record('S14 Person × instrument-period fixed effects (no 2021→2022 within-person change used)', f, b, V, d, panel='c')
pids = {g: set(d.PID) for g, d in G.items()}; shared = {p for a in pids for b_ in pids if a != b_ for p in pids[a] & pids[b_]}
run(f'S15 Excluding the {len(shared)} persons observed in two groups', B[~B.PID.isin(shared)], panel='c')
# (d) measurement
d8 = shr(base_all[(base_all.Year >= 2022) & base_all.House_h.notna() & base_all.total8_h.between(12, 28)])
run('S16 Eight domains incl. housework (2022–2024)', d8, doms=DOMS[:-1] + ['House', 'Free'], tot='total8_h', panel='d')
run('S17 Without the total-hours control', B, tot='1', panel='d')
alt = shr(base_all).copy()                                        # upper-bound coding of the "less than one hour" band, window re-applied
for v in ['Work', 'Educ', 'Child', 'Elder', 'Free']: alt[f'{v}_h'] = np.where(alt[f'{v}_code'] == 2, 1.0, alt[f'{v}_h'])
alt['total_h'] = alt[[f'{d}_h' for d in DOMS]].sum(axis=1)
run('S18 Half-hour band coded as 1 h (window re-applied)', win(alt), panel='d')
# (e) context and inference
d = drop_singletons(panelize(B), ['PID', 'scale_year']); d['muni_year'] = d.muni + '_' + d.Year.astype(str); d = drop_singletons(d, ['PID', 'muni_year'])
f, b, V = fit(d, fe='PID + muni_year'); record('S19 Municipality × year fixed effects', f, b, V, d, persons=d.PID.nunique(), munis=d.muni.nunique(), panel='e')
run('S20 Standard errors clustered by person', B, vcov={'CRV1': 'PID'}, panel='e')
run('S21 Two-way clustering (municipality and person)', B, vcov={'CRV1': 'muni+PID'}, panel='e')
# definition of shrinking: a check of the group threshold, distinct from the tercile dose-response of Table S4
run('S22 Shrinking defined as ≤ −0.5%/yr', shr(win(base_all), -0.005), panel='e')
run('S23 Shrinking defined as ≤ −1.0%/yr', shr(win(base_all), -0.010), panel='e')

T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT_S, 'tableS3_sensitivity.csv'), index=False)
md = pd.DataFrame({'Specification': T.spec, 'Eldercare (+1 h)': T.elder_fmt, 'Sleep (+1 h)': T.sleep_fmt, 'Eldercare → sleep': T.elder_to_sleep_fmt, 'Person-years': [f'{n:,}' for n in T.N], 'Persons': [f'{n:,}' for n in T.persons]})
with open(os.path.join(OUT_S, 'tableS3_sensitivity.md'), 'w', encoding='utf-8') as fh:   # one block per panel
    for pn in dict.fromkeys(T.panel):
        fh.write(pn + '\n\n' + md_str(md[T.panel.values == pn]) + '\n')
    fh.write('Notes: Adults aged 15–64 in shrinking SMCs. Eldercare and sleep are deviation-coded domain values (one hour drawn evenly from the other six); eldercare → sleep is the pairwise reallocation; 95% intervals with municipality-clustered standard errors (S20: person; S21: municipality and person) and t(G − 1) critical values. S14 replaces the person effect by a person × period effect (2020–2021 / 2022–2024), so that no within-person change across the change of the time-use module is used. S19 drops person-years that are alone in a person or municipality-year cell after iterative removal. *** p < 0.001, ** p < 0.01, * p < 0.05.\n')
