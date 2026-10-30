"""Shared data preparation and model helpers (sample definition, Eq. (1), contrast vectors, inference).

Inputs
  panel.csv                         person-year panel (one row per PID x Year); layout in data/schema.csv
  data/large_cities.csv             the 105 statutory large cities (designated, core and former special cities)
  data/municipal_means_2018_2023.csv  Basic Resident Register population, in- and out-migration, 2018-2023 means
  data/municipal_annual.csv         (optional) the same series by municipality x year; used instead of the means if present

Every analysis script imports `load_panel()` and the estimation helpers below, so the sample definition and the
model are written in one place only.

Paths: the panel file and the output directory are taken from the environment variables TSQ_PANEL and TSQ_OUTDIR
(set by run_all.py from its --panel and --outdir arguments); the defaults are data/panel.csv and output/.
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings('ignore')
import pyfixest as pf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
PANEL_PATH = os.environ.get('TSQ_PANEL', os.path.join(DATA, 'panel.csv'))
OUT = os.environ.get('TSQ_OUTDIR', os.path.join(ROOT, 'output'))
OUT_T = os.path.join(OUT, 'main'); OUT_S = os.path.join(OUT, 'supplement'); OUT_F = os.path.join(OUT, 'figures')
for _d in (OUT_T, OUT_S, OUT_F): os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- constants
DOMS = ['Sleep', 'Educ', 'Commute', 'Work', 'Child', 'Elder', 'Free']          # seven time-use domains; Free = omitted
HOURS = [f'{d}_h' for d in DOMS if d != 'Free']                                   # regressors (free time omitted)
RHS = ' + '.join(HOURS)
COV = ' + hhsize + spouse + has_child + child_preschool + coresid_parent + C(emp_grp)'   # time-varying covariates
EN = {'Sleep': 'Sleep', 'Educ': 'Study', 'Commute': 'Commuting', 'Work': 'Work', 'Child': 'Childcare', 'Elder': 'Eldercare', 'Free': 'Free time'}
SHRINK_THRESHOLD = -0.0015      # CAGR 2017-2023 <= -0.15 %/yr  -> shrinking SMC
TOTAL_WINDOW = (10, 26)         # plausible total of the seven domains, hours/day
YEARS = (2020, 2024)            # the 2025 wave uses a redesigned time-use module and is excluded

# column names of earlier builds of the panel that are still accepted
LEGACY_COLUMNS = {'Maeged_flag': 'merged_flag', 'Metropolice_flag': 'metro_flag'}


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ---------------------------------------------------------------- city size by the Local Autonomy Act
# Large cities = the 20 designated cities, 62 core cities and 23 former special cities (Ministry of Internal Affairs and
# Communications list, 2025: https://www.soumu.go.jp/main_sosiki/jichi_gyousei/bunken/chihou-koukyoudantai_kubun.html)
# plus the 23 special wards of Tokyo.  Everything else is a small or medium-sized city (SMC): ordinary cities, towns and
# villages.  The survey's own city-type code (kept as `scale_survey`) departs from the statutory list for eight
# municipalities and is not used for classification.
_LC = pd.read_csv(os.path.join(DATA, 'large_cities.csv'), encoding='utf-8-sig')
assert (_LC.category == 'designated').sum() == 20 and (_LC.category == 'core').sum() == 62 and (_LC.category == 'former_special').sum() == 23
LARGE_CITY_NAMES = list(_LC.name_ja)
TOKYO_WARDS = [f'131{i:02d}' for i in range(1, 24)]


def read_panel(path=PANEL_PATH):
    """Read the panel CSV, zero-pad the 5-digit municipality code and accept legacy column names."""
    P = pd.read_csv(path, encoding='utf-8-sig', low_memory=False).rename(columns=LEGACY_COLUMNS)
    P['muni'] = P['CityCode-5'].apply(lambda v: f'{int(v):05d}' if pd.notna(v) else np.nan)
    return P


def classify_scale(P):
    """'Large' for the 105 statutory large cities (matched on the municipality name, '<name>市', so that a city-level or a
    ward-level record of a designated city is caught) and the 23 Tokyo wards (matched on the code); 'SMC' for every other
    identified municipality; NaN where the municipality is unknown."""
    name = P.CityName.fillna('').astype(str)
    pat = '^(' + '|'.join(LARGE_CITY_NAMES) + ')市'
    is_large = name.str.match(pat) | P.muni.isin(TOKYO_WARDS)
    P['scale_survey'] = P['scale'] if 'scale' in P.columns else np.nan
    P['scale'] = pd.Series(np.where(is_large, 'Large', 'SMC'), index=P.index, dtype=object).where(P.muni.notna())
    found = set(name[is_large & P.muni.notna()].str.extract(pat)[0].dropna())
    P.attrs['large_cities_missing'] = sorted(set(LARGE_CITY_NAMES) - found)
    P.attrs['large_municipalities'] = int(P.loc[P.scale == 'Large', 'muni'].nunique())
    return P


def load_panel():
    """Step 1 of the algorithm: read the panel, derive the analysis variables, attach the municipal context.
    Returns the full panel (all years) with LS0, muni, scale_year, cagr_z, shrinking, stayer and the moderator z-scores."""
    P = classify_scale(read_panel())
    # household variables: one-person households have no co-residents by definition
    for c in ['hh_spouse', 'hh_child', 'hh_parent', 'child_preschool']:
        P[c] = np.where(P.hhsize == 1, 0, P[c])
    P['spouse'] = P.hh_spouse; P['has_child'] = P.hh_child; P['coresid_parent'] = P.hh_parent
    P['emp_grp'] = P.emp_grp.fillna('nonemp')
    # record-linkage check: age bands must not fall, nor rise by more than the elapsed years allow; sex must not change
    bad = age_inconsistent_pids(P)
    P.attrs['age_inconsistent_persons'] = int(len(bad)); P.attrs['age_inconsistent_rows'] = int(P.PID.isin(bad).sum())
    P = P[~P.PID.isin(bad)].copy()
    # residential stability, defined on every observed row of the person before any other exclusion
    P['stayer'] = P.groupby('PID').muni.transform(lambda m: m.dropna().nunique() <= 1)
    # usable records: complete seven-domain time use, life satisfaction, identified and classified municipality
    P = P[P.tu_complete.astype(bool) & P.LS.notna() & P.muni.notna() & P.scale.notna()].copy()
    P['LS0'] = P.LS - 1                                   # survey codes 1..11 -> 0..10
    P['scale_year'] = P.scale + '_' + P.Year.astype(str)  # city-size x year fixed effect
    # continuous context: CAGR standardised across municipalities (one value per municipality)
    mm = P.drop_duplicates('muni')
    P['cagr_z'] = (P.cagr_17_23 - mm.cagr_17_23.mean()) / mm.cagr_17_23.std()
    P['nmr_z'] = (P.nmr_mean_18_23 - mm.nmr_mean_18_23.mean()) / mm.nmr_mean_18_23.std()
    P['shrinking'] = (P.cagr_17_23 <= SHRINK_THRESHOLD).astype(int)
    attrs = dict(P.attrs); P = add_moderators(P); P.attrs.update(attrs)
    P.attrs['CAGR_MEAN'] = float(mm.cagr_17_23.mean()); P.attrs['CAGR_SD'] = float(mm.cagr_17_23.std())
    return P


def age_inconsistent_pids(P):
    """Linkage check: persons whose 5-year age band falls between observed waves, or rises by more bands than the
    elapsed years allow (ceil(years/5)), or whose reported sex changes -- signs of a broken panel link."""
    P = P.sort_values(['PID', 'Year'])
    dA = P.groupby('PID').Age.diff(); dY = P.groupby('PID').Year.diff()
    bad_age = P.loc[(dA < 0) | (dA > np.ceil(dY / 5)), 'PID'].unique()
    nsex = P.groupby('PID').Sex.nunique(); bad_sex = nsex[nsex > 1].index.values
    return np.union1d(bad_age, bad_sex)


def add_moderators(P):
    """Municipal indicators for the specificity checks, each oriented so that higher = more decline-prone,
    standardised across all municipalities in the Basic Resident Register file."""
    # either the annual file (code5, Year, Population, Inflow, Outflow) or its 2018-2023 means (code5, Pop, In, Out)
    f_annual = os.path.join(DATA, 'municipal_annual.csv'); f_means = os.path.join(DATA, 'municipal_means_2018_2023.csv')
    if os.path.exists(f_annual):
        pop = pd.read_csv(f_annual, encoding='utf-8-sig', dtype={'code5': str}); pop['code5'] = pop.code5.str.zfill(5)
        mu = pop[pop.Year.between(2018, 2023)].groupby('code5').agg(Pop=('Population', 'mean'), In=('Inflow', 'mean'), Out=('Outflow', 'mean'))
    else:
        mu = pd.read_csv(f_means, encoding='utf-8-sig', dtype={'code5': str}); mu['code5'] = mu.code5.str.zfill(5); mu = mu.set_index('code5')[['Pop', 'In', 'Out']]
    mu['D_netout'] = -(mu.In - mu.Out) / mu.Pop * 1000
    mu['D_outrate'] = mu.Out / mu.Pop * 1000
    mu['D_inrate_neg'] = -mu.In / mu.Pop * 1000
    mu['D_smallpop'] = -np.log(mu.Pop)
    area = P.drop_duplicates('muni').set_index('muni').Areakm2.astype(float)
    mu['D_sparse'] = -np.log(mu.Pop / area.reindex(mu.index))
    for m in ['D_netout', 'D_outrate', 'D_inrate_neg', 'D_smallpop', 'D_sparse']:
        mu[m + '_z'] = (mu[m] - mu[m].mean()) / mu[m].std()
    P = P.merge(mu[[c for c in mu.columns if c.endswith('_z')]], left_on='muni', right_index=True, how='left')
    P['D_merged_z'] = P.merged_flag.astype(float)
    P['D_nonmetro_z'] = 1 - P.metro_flag.astype(float)
    return P


def panelize(d):
    """Keep persons observed in at least two waves of the given data (within-person identification)."""
    n = d.groupby('PID').Year.transform('nunique')
    return d[n >= 2].copy()


def analytic_sample(P, years=YEARS, window=TOTAL_WINDOW):
    """Step 2: waves, plausible total hours, >= 2 waves."""
    d = P[P.Year.between(*years)]
    if window: d = d[d.total_h.between(*window)]
    return panelize(d)


def groups(d):
    """Step 3: the three adult groups (each re-panelized so that every person contributes within-group variation)."""
    a = d[d.age_grp == '15-64']
    return {
        'Shrinking SMCs': panelize(a[(a.scale == 'SMC') & (a.shrinking == 1)]),
        'Non-shrinking SMCs': panelize(a[(a.scale == 'SMC') & (a.shrinking == 0)]),   # CAGR above the threshold
        'Large cities': panelize(a[a.scale == 'Large']),
    }


# ---------------------------------------------------------------- models
def fit(sub, mod=None, cov=COV, tot='total_h', rhs=RHS, vcov=None, fe='PID + scale_year'):
    """Isotemporal substitution model (Eq. 1): LS0 ~ hours(6 domains, free time omitted) + total
    + age bands + covariates | person FE + city-size x year FE; SE clustered by municipality.
    With a moderator, every hours term AND the total are interacted ((hours + total) x mod), so that the
    deviation-coded interactions do not depend on which domain is omitted."""
    vcov = vcov or {'CRV1': 'muni'}
    x = f'({rhs} + {tot})*{mod}' if mod else f'{rhs} + {tot}'
    f = pf.feols(f"LS0 ~ {x} + C(Age){cov} | {fe}", data=sub, vcov=vcov)
    b = f.coef(); V = pd.DataFrame(f._vcov, index=b.index, columns=b.index)
    V.attrs['G'] = n_clusters(f)
    return f, b, V


def n_clusters(f):
    """Number of clusters used for t(G-1) inference, taken from the estimation sample of the fitted model
    (after pyfixest's NA and singleton removal; the smallest dimension for multi-way clustering)."""
    return int(min(f._G)) if getattr(f, '_is_clustered', False) else None


def drop_singletons(d, fes):
    """Iteratively drop observations that are alone in any fixed-effect cell (what pyfixest does internally),
    so that persons and municipalities can be counted on the estimation sample."""
    d = d.copy()
    while True:
        n0 = len(d)
        for fe in fes: d = d[d.groupby(fe)[fe].transform('size') > 1]
        if len(d) == n0: return d


def dev(names, X, mod=None, doms=DOMS):
    """Contrast vector for the deviation-coded domain value (Eq. 3): +1 h in X taken evenly (1/6 each) from the other six."""
    c = pd.Series(0.0, index=names); suf = '' if mod is None else f':{mod}'; k = len(doms) - 1
    if X != 'Free': c[f'{X}_h{suf}'] += 1
    for Y in doms:
        if Y == X or Y == 'Free': continue
        c[f'{Y}_h{suf}'] -= 1 / k
    return c


def sub_c(names, A, B, mod=None, x=0):
    """Contrast vector for the reallocation of 1 h from A to B (Eq. 2: beta_B - beta_A, beta_Free = 0),
    evaluated at moderator value x when the model is interacted."""
    c = pd.Series(0.0, index=names)
    for suf, k in [('', 1)] + ([(f':{mod}', x)] if mod else []):
        if B != 'Free': c[f'{B}_h{suf}'] += k
        if A != 'Free': c[f'{A}_h{suf}'] -= k
    return c


def est(c, b, V, G=None):
    """Point estimate, SE and p of a linear combination c'b with Var = c'Vc (Eq. 4). p uses t(G-1) when the number
    of clusters G is known (as in pyfixest), otherwise the normal approximation."""
    from scipy.stats import t as tdist
    e = float(c @ b); se = float(np.sqrt(c @ V.values @ c)); G = G or V.attrs.get('G')
    p = 2 * (1 - tdist.cdf(abs(e / se), G - 1)) if G else 2 * (1 - norm.cdf(abs(e / se)))
    return e, se, p


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(p), method='fdr_bh')[1]


def crit(G=None):
    """95% critical value: t(G-1) quantile when the number of clusters G is known, else 1.96."""
    from scipy.stats import t as tdist
    return float(tdist.ppf(0.975, G - 1)) if G else 1.96


def fmt(e, se, p, G=None):
    c = crit(G); return f"{e:+.3f}{stars(p)} [{e - c * se:+.3f}, {e + c * se:+.3f}]".replace('-', '−')


def md_str(df):
    """A DataFrame as a markdown pipe table (string)."""
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r.values) + ' |')
    return '\n'.join(lines) + '\n'


def to_md(df, path, floatfmt=None):
    """Write a DataFrame as a markdown pipe table (used for the manuscript tables)."""
    with open(path, 'w', encoding='utf-8') as f: f.write(md_str(df))


def bold(s):
    """Mark a table cell as significant after the false-discovery correction: whole cell in bold, inner stars escaped."""
    return '**' + s.replace('*', '\\*') + '**'
