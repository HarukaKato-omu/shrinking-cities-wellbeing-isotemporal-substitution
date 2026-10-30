"""Steps 1-3: sample construction. Outputs Table S1 (sample flow and the person-years excluded by the total-hours
window) to output/supplement/ and Table 1 (descriptives of the three estimation samples) to output/main/."""
import os, numpy as np, pandas as pd
from common import *

# ---- Table S1: sample flow, counted on the raw file so that every exclusion is visible
raw = classify_scale(read_panel())
flow = []
def step(label, d): flow.append(dict(step=label, person_years=len(d), persons=d.PID.nunique(), municipalities=d['CityCode-5'].nunique()))
step('Rebuilt panel, 2020–2025', raw)
d = raw[raw.Year.between(*YEARS)]; step('Drop the 2025 wave (different instrument)', d)
bad = age_inconsistent_pids(raw); d = d[~d.PID.isin(bad)]; step(f'Drop {len(bad)} persons whose age band or sex is inconsistent across waves (linkage check)', d)
d = d[d.tu_complete.astype(bool)]; step('Complete answers to the seven time-use items', d)
d = d[d.LS.notna()]; step('Life satisfaction reported', d)
d = d[d['CityCode-5'].notna() & d.scale.notna()]; step('Municipality identified and classified', d)
dw = d.copy(); d = d[d.total_h.between(*TOTAL_WINDOW)]; step('Total of the seven domains within 10–26 h', d)
# who is excluded by the window? (Table S1, panel b)
rows = []
for lab, x in [('Kept (10-26 h)', dw[dw.total_h.between(*TOTAL_WINDOW)]), ('Excluded: total < 10 h', dw[dw.total_h < 10]), ('Excluded: total > 26 h', dw[dw.total_h > 26])]:
    rows.append(dict(group=lab, person_years=len(x), sleep=x.Sleep_h.mean(), work=x.Work_h.mean(), free=x.Free_h.mean(), eldercare=x.Elder_h.mean(), any_eldercare_pct=100 * (x.Elder_h > 0).mean(), eldercare_ge8h_pct=100 * (x.Elder_h >= 8).mean(), LS=(x.LS - 1).mean(), share_2020_21_pct=100 * x.Year.le(2021).mean()))
pd.DataFrame(rows).round(2).to_csv(os.path.join(OUT_S, 'tableS1b_window_exclusions.csv'), index=False); print(pd.DataFrame(rows).round(2).to_string(index=False))
n = d.groupby('PID').Year.transform('nunique'); d = d[n >= 2]; step('Observed in at least two waves', d)
a = d[d.age_grp == '15-64']; step('Adults aged 15–64', a)
n = a.groupby('PID').Year.transform('nunique'); a2 = a[n >= 2]; step('Adults observed at least twice as adults', a2)
a2 = a2.assign(grp=np.where(a2.scale == 'Large', 'Large', np.where(a2.cagr_17_23 <= SHRINK_THRESHOLD, 'SMC-shr', 'SMC-grw')))
n = a2.groupby(['PID', 'grp']).Year.transform('nunique'); a3 = a2[n >= 2]; step('Observed at least twice within the same group (estimation samples)', a3)
flow[-1]['note'] = f"persons in two groups: {int((a3.groupby('PID').grp.nunique() > 1).sum())}"
pd.DataFrame(flow).to_csv(os.path.join(OUT_S, 'tableS1_flow.csv'), index=False)

# Table S1 as one table with two panels: (a) the flow, (b) the person-years excluded by the window
fa = pd.DataFrame(flow)
md_a = md_str(pd.DataFrame({'Step': fa.step, 'Person-years': [f'{v:,}' for v in fa.person_years], 'Persons': [f'{v:,}' for v in fa.persons], 'Municipalities': [f'{v:,}' for v in fa.municipalities]}))
fb = pd.DataFrame(rows)
md_b = md_str(pd.DataFrame({'': fb.group.str.replace('Kept (10-26 h)', 'Retained (10–26 h)', regex=False).str.replace('Excluded: total', 'Excluded, total', regex=False),
                            'Person-years': [f'{v:,}' for v in fb.person_years], 'Sleep, h': fb.sleep.map('{:.2f}'.format), 'Work, h': fb.work.map('{:.2f}'.format), 'Free time, h': fb.free.map('{:.2f}'.format),
                            'Eldercare, h': fb.eldercare.map('{:.2f}'.format), 'Any eldercare, %': fb.any_eldercare_pct.map('{:.1f}'.format), 'Eldercare ≥ 8 h, %': fb.eldercare_ge8h_pct.map('{:.1f}'.format),
                            'Life satisfaction (0–10)': fb.LS.map('{:.2f}'.format), '2020–2021 waves, %': fb.share_2020_21_pct.map('{:.1f}'.format)}))
with open(os.path.join(OUT_S, 'tableS1_sample_construction.md'), 'w', encoding='utf-8') as fh:
    fh.write('(a) Person-years, persons and municipalities retained at each step\n\n' + md_a
             + '\nThe last row is the sum of the three estimation samples of Table 1; ' + flow[-1]['note'] + '.\n\n'
             + '(b) Person-years excluded by the total-hours window, compared with the retained sample (means over person-years at the step before the window)\n\n' + md_b)

# ---- Table 1 on the exact estimation samples
P = load_panel(); S = analytic_sample(P); G = groups(S)
rows = []
def row(lab, f, fmt='{:.2f}'): rows.append([lab] + [fmt.format(f(g)) for g in G.values()])
def head(lab): rows.append([f'**{lab}**'] + [''] * len(G))                      # category rows
head('Sample'); row('Person-years', lambda d: len(d), '{:,.0f}'); row('Persons', lambda d: d.PID.nunique(), '{:,.0f}')
row('Waves per person', lambda d: len(d) / d.PID.nunique()); row('Municipalities', lambda d: d.muni.nunique(), '{:,.0f}')
head('Municipal context'); row('Population change 2017–2023, %/yr (municipality mean)', lambda d: d.drop_duplicates('muni').cagr_17_23.mean() * 100, '{:+.2f}'); rows[-1] = [rows[-1][0]] + [v.replace('-', '−') for v in rows[-1][1:]]
head('Life satisfaction (0–10)'); rows.append(['Mean (SD)'] + [f'{d.LS0.mean():.2f} ({d.LS0.std():.2f})' for d in G.values()])
row('Within-person SD', lambda d: (d.LS0 - d.groupby('PID').LS0.transform('mean')).std())
head('Time use, hours per day')
for v, l in [('Sleep_h', 'Sleep'), ('Work_h', 'Work'), ('Commute_h', 'Commuting'), ('Child_h', 'Childcare'), ('Elder_h', 'Eldercare'), ('Educ_h', 'Study'), ('Free_h', 'Free time'), ('total_h', 'Total of the seven domains')]:
    row(l, lambda d, v=v: d[v].mean())
head('Eldercare exposure'); row('Any eldercare, % of person-years', lambda d: (d.Elder_h > 0).mean() * 100, '{:.1f}')
row('Persons whose eldercare hours change between waves', lambda d: (d.groupby('PID').Elder_h.nunique() > 1).sum(), '{:,.0f}')
head('Demographics, %'); row('Female', lambda d: (d.Sex == 2).mean() * 100, '{:.1f}')
row('Aged 15–39', lambda d: d.Age.isin([1, 2, 3, 4, 5]).mean() * 100, '{:.1f}')
head('Employment status, %')
for lab, cats in [('Regular employee', ['regular']), ('Non-regular employee', ['nonregular']), ('Self-employed or executive', ['self', 'exec']), ('Student', ['student']), ('Not employed or home-based worker', ['nonemp', 'homework'])]:
    row(lab, lambda d, cats=cats: d.emp_grp.isin(cats).mean() * 100, '{:.1f}')
head('Household'); row('Household size, persons', lambda d: d.hhsize.mean())
for lab, v in [('Lives with spouse, %', 'spouse'), ('Lives with child, %', 'has_child'), ('Preschool child in household, %', 'child_preschool'), ('Lives with parent, %', 'coresid_parent')]:
    row(lab, lambda d, v=v: d[v].mean() * 100, '{:.1f}')
T1 = pd.DataFrame(rows, columns=[''] + list(G))
T1.to_csv(os.path.join(OUT_T, 'table1_sample.csv'), index=False); to_md(T1, os.path.join(OUT_T, 'table1_sample.md'))
print(pd.DataFrame(flow).to_string(index=False)); print(T1.to_string(index=False))
print('CAGR across municipalities: mean %.2f %%/yr, SD %.2f' % (P.attrs['CAGR_MEAN'] * 100, P.attrs['CAGR_SD'] * 100))
print('large-city municipalities in the panel:', P.attrs['large_municipalities'], '| statutory large cities not found:', P.attrs['large_cities_missing'] or 'none')
