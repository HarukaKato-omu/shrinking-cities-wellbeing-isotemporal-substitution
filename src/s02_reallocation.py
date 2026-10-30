"""Steps 4-5: the reallocation model (Eq. 1) and the specificity checks.
Outputs (Table 2 to output/main/, the rest to output/supplement/):
  table2_domain_values.csv/.md     deviation-coded domain values (Eq. 3), three groups + contrasts with non-shrinking SMCs
                                   and large cities (the CSV also holds the 2022-2024 replication = Table S3, S12)
  bh_familywide.csv                Benjamini-Hochberg over the 21 domain values of all three groups (note to Section 4.1)
  tableS2_reallocations.csv/.md    21 pairwise reallocations (Eq. 2), three groups, BH-adjusted p for shrinking SMCs
  matrix_full.csv                  all 42 directed reallocations per group (Fig. 2)
  spec_gradient.csv, me_grid.csv   continuous CAGR interaction: per-SD interaction, implied values at -1/+1 SD, and the
                                   domain values along the CAGR axis (Fig. 3)
  spec_terciles.csv                domain values by CAGR tercile of SMC municipalities (Table S4, panel b)
  tableS4_moderators.csv/.md       7 domains x 5 indicators interaction grid, SMC adults and large-city adults (Fig. S1)
  tableS4b_joint.csv/.md           joint model: population decline, density and size entered together
  tableS4c_threeway.csv/.md        domain x CAGR x city-size three-way interactions
  tableS4_eldercare_indicators.md  manuscript Table S4: (a) the eldercare value x each indicator, (b) by tercile
"""
import os, numpy as np, pandas as pd
from common import *

P = load_panel(); S = analytic_sample(P); G = groups(S)
SA = analytic_sample(P, years=(2022, 2024)); GA = groups(SA)       # replication after the 2022 instrument change
fits = {g: fit(d) for g, d in G.items()}

# ---------------------------------------------------------------- Table 2: domain values
rows = []
for X in DOMS:
    r = {'Domain': EN[X]}
    for g, (f, b, V) in fits.items():
        e, se, p = est(dev(list(b.index), X), b, V); r[g] = (e, se, p)
    f, b, V = fit(GA['Shrinking SMCs']); r['Replication 2022-2024 (shrinking SMCs)'] = est(dev(list(b.index), X), b, V)
    rows.append(r)
T2 = pd.DataFrame(rows)
sh = np.array([r['Shrinking SMCs'][2] for r in rows]); T2['adj_p_shrinking'] = bh(sh)
d = [(r['Shrinking SMCs'][0] - r['Non-shrinking SMCs'][0], np.sqrt(r['Shrinking SMCs'][1] ** 2 + r['Non-shrinking SMCs'][1] ** 2)) for r in rows]
from scipy.stats import t as tdist
Gmin = min(fits['Shrinking SMCs'][2].attrs['G'], fits['Non-shrinking SMCs'][2].attrs['G'])
T2['Difference shrinking - non-shrinking'] = [(e, se, 2 * (1 - tdist.cdf(abs(e / se), Gmin - 1))) for e, se in d]   # t with the smaller cluster count
dL = [(r['Shrinking SMCs'][0] - r['Large cities'][0], np.sqrt(r['Shrinking SMCs'][1] ** 2 + r['Large cities'][1] ** 2)) for r in rows]
GminL = min(fits['Shrinking SMCs'][2].attrs['G'], fits['Large cities'][2].attrs['G'])
T2['Difference shrinking - large'] = [(e, se, 2 * (1 - tdist.cdf(abs(e / se), GminL - 1))) for e, se in dL]
GA_shr = fit(GA['Shrinking SMCs'])
Gcol = {'Shrinking SMCs': fits['Shrinking SMCs'][2].attrs['G'], 'Non-shrinking SMCs': fits['Non-shrinking SMCs'][2].attrs['G'], 'Large cities': fits['Large cities'][2].attrs['G'],
        'Difference shrinking - non-shrinking': Gmin, 'Difference shrinking - large': GminL, 'Replication 2022-2024 (shrinking SMCs)': GA_shr[2].attrs['G']}
long = []
for _, r in T2.iterrows():
    for col, Gc in Gcol.items():
        e, se, p = r[col]; long.append(dict(domain=r.Domain, column=col, est=e, se=se, p=p, crit=crit(Gc), lo=e - crit(Gc) * se, hi=e + crit(Gc) * se, stars=stars(p), adj_p=r.adj_p_shrinking if col == 'Shrinking SMCs' else np.nan))
pd.DataFrame(long).to_csv(os.path.join(OUT_T, 'table2_domain_values.csv'), index=False)
# manuscript Table 2: no adjusted-p column (FDR-significant cells in bold) and no replication column (= Table S3, S12)
md = pd.DataFrame({'Domain (+1 h, taken evenly from the other six)': T2.Domain,
                   'Shrinking SMCs': [bold(fmt(*v, G=Gcol['Shrinking SMCs'])) if q < 0.05 else fmt(*v, G=Gcol['Shrinking SMCs']) for v, q in zip(T2['Shrinking SMCs'], T2.adj_p_shrinking)],
                   'Non-shrinking SMCs': [fmt(*v, G=Gcol['Non-shrinking SMCs']) for v in T2['Non-shrinking SMCs']],
                   'Large cities': [fmt(*v, G=Gcol['Large cities']) for v in T2['Large cities']],
                   'Difference (shrinking − non-shrinking)': [fmt(*v, G=Gmin) for v in T2['Difference shrinking - non-shrinking']],
                   'Difference (shrinking − large)': [fmt(*v, G=GminL) for v in T2['Difference shrinking - large']]})
md.loc[len(md)] = ['Person-years', f"{fits['Shrinking SMCs'][0]._N:,}", f"{fits['Non-shrinking SMCs'][0]._N:,}", f"{fits['Large cities'][0]._N:,}", '—', '—']
NOTE2 = ('\nBold: significant after Benjamini–Hochberg correction at a false discovery rate of 5% over the seven domain values of the shrinking-SMC model (adjusted p: '
         + ', '.join(f'{d} {q:.3f}' for d, q in zip(T2.Domain, T2.adj_p_shrinking)) + '). The re-estimation on the 2022–2024 waves is Table S3, S12.\n')
# family-wide BH over the 21 domain values of the three groups, for the note in Section 4.1
fw = [(g, r['Domain'], r[g][2]) for r in rows for g in G]
FW = pd.DataFrame(fw, columns=['group', 'domain', 'p']); FW['adj_p_familywide'] = bh(FW.p); FW.to_csv(os.path.join(OUT_S, 'bh_familywide.csv'), index=False)
print('family-wide BH, eldercare shrinking:', FW[(FW.group == 'Shrinking SMCs') & (FW.domain == 'Eldercare')].adj_p_familywide.round(3).iloc[0])
to_md(md, os.path.join(OUT_T, 'table2_domain_values.md')); open(os.path.join(OUT_T, 'table2_domain_values.md'), 'a', encoding='utf-8').write(NOTE2); print(md.to_string(index=False))

# ---------------------------------------------------------------- Table S2: 21 reallocations
order = ['Work', 'Commute', 'Child', 'Elder', 'Educ', 'Sleep', 'Free']; rows = []
for i, A in enumerate(order):
    for B in order[i + 1:]:
        r = {'from': A, 'to': B, 'Reallocation': f'{EN[A]} → {EN[B]}'}
        for g, (f, b, V) in fits.items(): r[g] = est(sub_c(list(b.index), A, B), b, V)
        rows.append(r)
T3 = pd.DataFrame(rows); T3['adj_p_shrinking'] = bh([v[2] for v in T3['Shrinking SMCs']])
long = [dict(frm=r['from'], to=r['to'], column=g, est=r[g][0], se=r[g][1], p=r[g][2], crit=crit(Gcol[g]), stars=stars(r[g][2]), adj_p=r.adj_p_shrinking if g == 'Shrinking SMCs' else np.nan) for _, r in T3.iterrows() for g in G]
pd.DataFrame(long).to_csv(os.path.join(OUT_S, 'tableS2_reallocations.csv'), index=False)
md = pd.DataFrame({'Reallocation (1 h from X to Y)': T3.Reallocation, 'Shrinking SMCs': [bold(fmt(*v, G=Gcol['Shrinking SMCs'])) if q < 0.05 else fmt(*v, G=Gcol['Shrinking SMCs']) for v, q in zip(T3['Shrinking SMCs'], T3.adj_p_shrinking)],
                   'Non-shrinking SMCs': [fmt(*v, G=Gcol['Non-shrinking SMCs']) for v in T3['Non-shrinking SMCs']], 'Large cities': [fmt(*v, G=Gcol['Large cities']) for v in T3['Large cities']]})   # bold = FDR 5%
to_md(md, os.path.join(OUT_S, 'tableS2_reallocations.md'))
qs = T3.adj_p_shrinking[T3.adj_p_shrinking < 0.05]
fdr_note = f'adjusted p from {qs.min():.3f} to {qs.max():.3f}' if len(qs) else 'no reallocation significant after correction'
open(os.path.join(OUT_S, 'tableS2_reallocations.md'), 'a', encoding='utf-8').write(f'\nBold: significant after Benjamini–Hochberg correction at a false discovery rate of 5% over the 21 reallocations of the shrinking-SMC model ({fdr_note}). Person-years: ' + '; '.join(f"{fits[g][0]._N:,}" for g in G) + '.\n')
print(md.to_string(index=False))
# full 7x7 matrices (both directions) for Fig. 2
full = [dict(group=g, frm=A, to=B, est=e, se=se, p=p) for g, (f, b, V) in fits.items() for A in DOMS for B in DOMS if A != B for e, se, p in [est(sub_c(list(b.index), A, B), b, V)]]
pd.DataFrame(full).to_csv(os.path.join(OUT_S, 'matrix_full.csv'), index=False)

# ---------------------------------------------------------------- the two adult strata used by the specificity checks
# The analysis is restricted to adults aged 15-64 at the sampling stage (from 2022 the time-use module was asked only
# of respondents in work or education, so the 65+ stratum is small and selective).
strata = {k: panelize(v) for k, v in {'SMC adults': S[(S.scale == 'SMC') & (S.age_grp == '15-64')], 'Large adults': S[(S.scale == 'Large') & (S.age_grp == '15-64')]}.items()}

# ---------------------------------------------------------------- Specificity (i): continuous gradient among all SMC adults
smc = strata['SMC adults']; f2, b2, V2 = fit(smc, 'cagr_z'); rows = []; grid = []
for X in DOMS:
    cm = dev(list(b2.index), X); ci = dev(list(b2.index), X, 'cagr_z')
    for lab, c in [('interaction_perSD', ci), ('at_minus1SD', cm - ci), ('at_mean', cm), ('at_plus1SD', cm + ci)]:
        e, se, p = est(c, b2, V2); rows.append(dict(domain=EN[X], stat=lab, est=e, se=se, p=p, stars=stars(p), N=f2._N))
    for z in np.arange(-3.5, 1.51, 0.125):                      # CAGR in %/yr -> z
        zc = (z / 100 - P.attrs['CAGR_MEAN']) / P.attrs['CAGR_SD']; e, se, p = est(cm + zc * ci, b2, V2)
        grid.append(dict(domain=EN[X], cagr=z, est=e, se=se, lo=e - crit(V2.attrs['G']) * se, hi=e + crit(V2.attrs['G']) * se))   # t(G-1) band
pd.DataFrame(rows).to_csv(os.path.join(OUT_S, 'spec_gradient.csv'), index=False); pd.DataFrame(grid).to_csv(os.path.join(OUT_S, 'me_grid.csv'), index=False)
print(pd.DataFrame(rows).query("domain in ['Eldercare','Sleep']").round(3).to_string(index=False))
# terciles of SMC municipalities
q = smc.drop_duplicates('muni').cagr_17_23.quantile([1 / 3, 2 / 3]).values
smc = smc.assign(ter=np.where(smc.cagr_17_23 <= q[0], 'T1 fast', np.where(smc.cagr_17_23 <= q[1], 'T2 mid', 'T3 slow'))); rows = []
for t in ['T1 fast', 'T2 mid', 'T3 slow']:
    d = panelize(smc[smc.ter == t]); f, b, V = fit(d)
    for X in DOMS:
        e, se, p = est(dev(list(b.index), X), b, V); cv = crit(V.attrs['G']); rows.append(dict(tercile=t, mean_cagr_pct=d.drop_duplicates('muni').cagr_17_23.mean() * 100, domain=EN[X], est=e, se=se, p=p, crit=cv, lo=e - cv * se, hi=e + cv * se, stars=stars(p), N=f._N, munis=V.attrs['G']))
pd.DataFrame(rows).to_csv(os.path.join(OUT_S, 'spec_terciles.csv'), index=False)

# ---------------------------------------------------------------- Specificity (ii): moderator grid and joint model
# population decline versus three dimensions of peripherality (size, density, non-metropolitan location) and merger
MODS = [('cagr_z', 'Population decline (−CAGR 2017–23)', -1),
        ('D_smallpop_z', 'Population size (small)', 1), ('D_sparse_z', 'Density (low)', 1), ('D_nonmetro_z', 'Non-metropolitan', 1), ('D_merged_z', 'Merged municipality', 1)]
rows = []
for g in ['SMC adults', 'Large adults']:
    for m, lab, sg in MODS:
        d = strata[g].dropna(subset=[m]); f, b, V = fit(d, m)
        for X in DOMS:
            e, se, p = est(dev(list(b.index), X, m), b, V); rows.append(dict(group=g, moderator=m, label=lab, orientation=sg, domain=EN[X], est=e, est_oriented=sg * e, se=se, p=p, stars=stars(p), N=f._N, clusters=V.attrs['G'], crit=crit(V.attrs['G'])))
MOD = pd.DataFrame(rows); MOD.to_csv(os.path.join(OUT_S, 'tableS4_moderators.csv'), index=False)
# full moderator grids (one block per group; shown as Fig. S1) (SE, p and intervals in the CSV)
MOD['cell'] = [f'{v:+.3f}{st}'.replace('-', '−') for v, st in zip(MOD.est_oriented, MOD.stars)]
N_STRATA = {g: int(MOD[MOD.group == g].N.iloc[0]) for g in ['SMC adults', 'Large adults']}
with open(os.path.join(OUT_S, 'tableS4_moderators.md'), 'w', encoding='utf-8') as fh:
    for g, ttl in [('SMC adults', f"(a) SMC adults ({N_STRATA['SMC adults']:,} person-years)"), ('Large adults', f"(b) Large-city adults ({N_STRATA['Large adults']:,} person-years)")]:
        w = MOD[MOD.group == g].pivot(index='domain', columns='label', values='cell').reindex([EN[X] for X in DOMS])[[lab for _, lab, _ in MODS]]
        w.columns = [c.replace('Population decline (−CAGR 2017–23)', 'Population decline (−CAGR)') for c in w.columns]; w = w.reset_index().rename(columns={'domain': 'Domain'})
        fh.write(ttl + '\n\n' + md_str(w) + '\n')
    fh.write('Notes: Each cell is the change in the deviation-coded domain value per +1 SD of the indicator (0 → 1 for the two binary indicators), from a model in which every domain coefficient and the total-hours term are interacted with the indicator; indicators are oriented so that higher values denote the more decline-prone side (the CAGR column is sign-reversed). Exploratory, not adjusted for multiple testing. The merged-municipality cells rest on a handful of respondents whose eldercare hours change (see tableS4_eldercare_indicators.md) and are not interpreted. Standard errors and intervals are in tableS4_moderators.csv. *** p < 0.001, ** p < 0.01, * p < 0.05.\n')
rows = []
for g in ['SMC adults', 'Large adults']:
    d = strata[g].dropna(subset=['cagr_z', 'D_sparse_z', 'D_smallpop_z'])   # joint model = population decline versus density and size
    f = pf.feols(f"LS0 ~ ({RHS} + total_h)*cagr_z + ({RHS} + total_h)*D_sparse_z + ({RHS} + total_h)*D_smallpop_z + C(Age){COV} | PID + scale_year", data=d, vcov={'CRV1': 'muni'})
    b = f.coef(); V = pd.DataFrame(f._vcov, index=b.index, columns=b.index); V.attrs['G'] = n_clusters(f)
    for m, sg in [('cagr_z', -1), ('D_sparse_z', 1), ('D_smallpop_z', 1)]:
        for X in DOMS:
            e, se, p = est(dev(list(b.index), X, m), b, V); rows.append(dict(group=g, moderator=m, orientation=sg, domain=EN[X], est=e, est_oriented=sg * e, se=se, p=p, stars=stars(p), N=f._N, clusters=V.attrs['G'], crit=crit(V.attrs['G'])))
J = pd.DataFrame(rows); J.to_csv(os.path.join(OUT_S, 'tableS4b_joint.csv'), index=False)
JL = {'cagr_z': 'Population decline (−CAGR)', 'D_sparse_z': 'Density (low)', 'D_smallpop_z': 'Population size (small)'}
with open(os.path.join(OUT_S, 'tableS4b_joint.md'), 'w', encoding='utf-8') as fh:
    for g, ttl in [('SMC adults', '(a) SMC adults'), ('Large adults', '(b) Large-city adults')]:
        w = pd.DataFrame({'Domain': [EN[X] for X in DOMS], **{JL[m]: [fmt(r.est_oriented, r.se, r.p, G=int(r.clusters)) for _, r in J[(J.group == g) & (J.moderator == m)].iterrows()] for m in JL}})
        fh.write(ttl + f" ({int(J[J.group == g].N.iloc[0]):,} person-years)\n\n" + md_str(w) + '\n')
    fh.write('Notes: Interaction of each domain value with population decline (−CAGR 2017–2023), density and population size entered jointly (each oriented so that higher values denote the more decline-prone side; per +1 SD); 95% t(G−1) intervals with municipality-clustered standard errors. *** p < 0.001, ** p < 0.01, * p < 0.05.\n')

# ---------------------------------------------------------------- Specificity (iii): three-way interaction with city size
a = S[S.age_grp == '15-64'].copy(); a['smc'] = (a.scale == 'SMC').astype(int)
f = pf.feols(f"LS0 ~ ({RHS} + total_h)*cagr_z*smc + C(Age){COV} | PID + scale_year", data=a, vcov={'CRV1': 'muni'})
b = f.coef(); V = pd.DataFrame(f._vcov, index=b.index, columns=b.index); V.attrs['G'] = n_clusters(f); rows = []
for X in DOMS:
    c = pd.Series(0.0, index=b.index)
    if X != 'Free': c[f'{X}_h:cagr_z:smc'] += 1
    for Y in DOMS:
        if Y == X or Y == 'Free': continue
        c[f'{Y}_h:cagr_z:smc'] -= 1 / 6
    e, se, p = est(c, b, V); rows.append(dict(domain=EN[X], diff_SMC_minus_Large_perSD=e, se=se, p=p, stars=stars(p), N=f._N, clusters=V.attrs['G'], crit=crit(V.attrs['G'])))
W = pd.DataFrame(rows); W.to_csv(os.path.join(OUT_S, 'tableS4c_threeway.csv'), index=False); print(W.round(3).to_string(index=False))
md6c = pd.DataFrame({'Domain': W.domain, 'Gradient in SMCs minus gradient in large cities (per +1 SD of CAGR)': [fmt(r.diff_SMC_minus_Large_perSD, r.se, r.p, G=int(r.clusters)) for _, r in W.iterrows()]})
to_md(md6c, os.path.join(OUT_S, 'tableS4c_threeway.md'))
# manuscript Table S4: the eldercare value's interaction with each indicator, both groups, plus the joint and three-way models
rows5 = []
for m, lab, sg in MODS:
    r_s = MOD[(MOD.group == 'SMC adults') & (MOD.moderator == m) & (MOD.domain == 'Eldercare')].iloc[0]; r_l = MOD[(MOD.group == 'Large adults') & (MOD.moderator == m) & (MOD.domain == 'Eldercare')].iloc[0]
    rows5.append([lab.replace('Population decline (−CAGR 2017–23)', 'Population decline (−CAGR 2017–2023)'), fmt(r_s.est_oriented, r_s.se, r_s.p, G=int(r_s.clusters)), fmt(r_l.est_oriented, r_l.se, r_l.p, G=int(r_l.clusters))])
for m, lab in JL.items():
    r_s = J[(J.group == 'SMC adults') & (J.moderator == m) & (J.domain == 'Eldercare')].iloc[0]; r_l = J[(J.group == 'Large adults') & (J.moderator == m) & (J.domain == 'Eldercare')].iloc[0]
    rows5.append([f'Joint model: {lab[0].lower() + lab[1:]}', fmt(r_s.est_oriented, r_s.se, r_s.p, G=int(r_s.clusters)), fmt(r_l.est_oriented, r_l.se, r_l.p, G=int(r_l.clusters))])
r_w = W[W.domain == 'Eldercare'].iloc[0]
rows5.append(['Three-way model: SMC gradient minus large-city gradient (per +1 SD of CAGR)', fmt(r_w.diff_SMC_minus_Large_perSD, r_w.se, r_w.p, G=int(r_w.clusters)), ''])
T5 = pd.DataFrame(rows5, columns=['Indicator (oriented so that higher = more decline-prone)', 'SMC adults', 'Large-city adults'])
# panel (b): the eldercare value by tercile of municipal CAGR
TER = pd.read_csv(os.path.join(OUT_S, 'spec_terciles.csv')); TER = TER[TER.domain == 'Eldercare']
TL = {'T1 fast': 'T1 (fastest decline)', 'T2 mid': 'T2', 'T3 slow': 'T3 (slowest decline)'}
T5b = pd.DataFrame({'Tercile of SMC municipalities by CAGR 2017–2023': [TL[t] for t in TER.tercile], 'Mean CAGR, %/yr': [f'{v:+.2f}'.replace('-', '−') for v in TER.mean_cagr_pct],
                    'Eldercare (+1 h)': [fmt(r.est, r.se, r.p, G=int(r.munis)) for _, r in TER.iterrows()], 'Person-years': [f'{int(n):,}' for n in TER.N], 'Municipalities': [f'{int(n):,}' for n in TER.munis]})
_m = strata['SMC adults']; _m = _m[_m.D_merged_z == 1]; N_MERGED = int((_m.groupby('PID').Elder_h.nunique() > 1).sum())   # counted on the data
WORD = {k: v for k, v in zip(range(1, 13), ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'])}
with open(os.path.join(OUT_S, 'tableS4_eldercare_indicators.md'), 'w', encoding='utf-8') as fh:
    fh.write('(a) Interactions of the eldercare value with municipal indicators\n\n' + md_str(T5) + '\n(b) Eldercare value by tercile of municipal CAGR, SMC adults\n\n' + md_str(T5b) + '\n')
    fh.write(f"Notes: Adults aged 15–64 (SMC adults {N_STRATA['SMC adults']:,} person-years; large-city adults {N_STRATA['Large adults']:,}). (a) Each entry is the change in the deviation-coded value of an eldercare hour (0–10 scale) per +1 SD of the indicator, or from 0 to 1 for the two binary indicators, from a model in which every domain coefficient and the total-hours term are interacted with that indicator; indicators are oriented so that higher values denote the more decline-prone side (the CAGR row is sign-reversed). The joint model enters population decline (−CAGR), density and population size together, so that the decline interaction is estimated with size and density held constant. The three-way model pools all adults ({int(r_w.N):,} person-years) and interacts domain hours with CAGR and an SMC indicator; the entry is the difference between the SMC and the large-city eldercare gradient. The merged-municipality row rests on {WORD.get(N_MERGED, N_MERGED)} respondents whose eldercare hours change and is not interpreted. The same interactions for all seven domains are shown in Fig. S1 and listed in the supplementary file tableS4_moderators.csv. (b) Eq. (1) estimated separately within each tercile of SMC municipalities (persons observed at least twice in the tercile). 95% intervals in brackets, municipality-clustered standard errors and t(G − 1) critical values. *** p < 0.001, ** p < 0.01, * p < 0.05.\n")
print(T5.to_string(index=False)); print(T5b.to_string(index=False))
open(os.path.join(OUT_S, 'tableS4c_threeway.md'), 'a', encoding='utf-8').write(f'\nNotes: Three-way interaction of domain hours (and total hours) × standardized CAGR 2017–2023 × SMC indicator, all adults aged 15–64 ({int(W.N.iloc[0]):,} person-years); each row is the deviation-coded difference between the SMC and the large-city gradient of that domain value. 95% t(G−1) intervals with municipality-clustered standard errors. *** p < 0.001, ** p < 0.01, * p < 0.05.\n')
