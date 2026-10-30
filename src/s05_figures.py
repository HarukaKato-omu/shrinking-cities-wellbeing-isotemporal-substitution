"""Step 8: manuscript figures, saved as PNG (300 dpi) and PDF (vector) in output/figures/.
Fig. 1 domain values | Fig. 2 reallocation matrices | Fig. 3 gradient of decline | Fig. 4 eldercare-onset event study |
Fig. S1 moderator interactions.
Reads only the tables written by s01-s04, so the figures can be regenerated without re-estimating."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from common import OUT_T, OUT_S, OUT_F, EN, DOMS, SHRINK_THRESHOLD, stars, load_panel, analytic_sample

plt.rcParams['font.family'] = 'DejaVu Sans'; plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a96', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 9.5})
SURF = '#ffffff'; INK = '#0b0b0b'; INK2 = '#52514e'; BLUE = '#2a78d6'; ORANGE = '#eb6834'; GREY2 = '#9a9a96'; GREY = '#c3c2b7'
GROUPS = ['Shrinking SMCs', 'Non-shrinking SMCs', 'Large cities']; TITLES = ['(a) Shrinking SMCs', '(b) Non-shrinking SMCs', '(c) Large cities']
cmap = LinearSegmentedColormap.from_list('div', ['#1f5fb8', '#9ec0ea', '#e9e8e3', '#f3b79c', '#c9471f'])   # blue = loss, red = gain
def save(fig, name, **kw):
    for ext in ['png', 'pdf']: fig.savefig(os.path.join(OUT_F, f'{name}.{ext}'), dpi=300, facecolor=SURF, **kw)
    plt.close(fig); print('saved', name)

# ---------------------------------------------------------------- Fig. 1 domain values
T2 = pd.read_csv(os.path.join(OUT_T, 'table2_domain_values.csv')); ORD = ['Sleep', 'Free time', 'Study', 'Childcare', 'Commuting', 'Work', 'Eldercare']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, facecolor=SURF); y = np.arange(len(ORD))[::-1]
for ax, g, ttl in zip(axes, GROUPS, TITLES):
    s = T2[T2.column == g].set_index('domain'); ax.set_facecolor(SURF); ax.axvline(0, color=GREY2, lw=1)
    for yi, X in zip(y, ORD):
        r = s.loc[X]; col = ORANGE if r.p < 0.05 else GREY2; cv = r.crit if 'crit' in s.columns else 1.96; lo, hi = r.est - cv * r.se, r.est + cv * r.se
        ax.hlines(yi, lo, hi, color=col, lw=2.2); ax.scatter(r.est, yi, s=46, color=col, zorder=3, edgecolor=SURF, linewidth=1.2)
        ax.text(hi + 0.012, yi, f'{r.est:+.3f}{stars(r.p)}', va='center', fontsize=8.5, color=INK2)   # label after the interval, never over it
    ax.set_title(f'{ttl}, adults', loc='left', fontsize=9.5, color=INK); ax.set_xlim(-0.5, 0.56); ax.set_xlabel('Change in life satisfaction per +1 h (points, 95% CI)', color=INK2, fontsize=8.5)
axes[0].set_yticks(y); axes[0].set_yticklabels(ORD, color=INK)
fig.legend(handles=[Line2D([0], [0], marker='o', color=ORANGE, lw=2.2, label='p < 0.05'), Line2D([0], [0], marker='o', color=GREY2, lw=2.2, label='not significant')], loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'fig1_domain_values')

# ---------------------------------------------------------------- Fig. 2 reallocation matrices
M = pd.read_csv(os.path.join(OUT_S, 'matrix_full.csv')); DM = ['Work', 'Commute', 'Child', 'Elder', 'Educ', 'Sleep', 'Free']; LAB = {'Sleep': 'Sleep', 'Educ': 'Study', 'Commute': 'Commute', 'Work': 'Work', 'Child': 'Childcare', 'Elder': 'Eldercare', 'Free': 'Free'}
def panel(ax, g, title, vmax=0.2):
    s = M[M.group == g]; Z = np.full((7, 7), np.nan); S = np.full((7, 7), '', dtype=object)
    for _, r in s.iterrows(): Z[DM.index(r.frm), DM.index(r.to)] = r.est; S[DM.index(r.frm), DM.index(r.to)] = stars(r.p)
    ax.set_facecolor(SURF); ax.imshow(np.clip(np.where(np.isnan(Z), 0, Z), -vmax, vmax), cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    for i in range(7):
        for j in range(7):
            if i == j: ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#ffffff')); continue
            v = Z[i, j]; st = S[i, j]; ax.text(j, i, f'{v:+.2f}{st}', ha='center', va='center', fontsize=7.4, color='white' if abs(v) > 0.6 * vmax else INK, fontweight='bold' if st else 'normal')
    ax.set_xticks(range(7)); ax.set_xticklabels([LAB[d] for d in DM], color=INK, fontsize=8.5, rotation=30, ha='right'); ax.set_yticks(range(7)); ax.set_yticklabels([LAB[d] for d in DM], color=INK, fontsize=8.5)
    ax.set_xlabel('Destination (Y, +1 h)', color=INK2, fontsize=8.5); ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    if '(a)' in title: ax.set_ylabel('Origin (X, −1 h)', color=INK2, fontsize=8.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURF, gridspec_kw={'wspace': 0.32})
for ax, g, ttl in zip(axes, GROUPS, TITLES): panel(ax, g, f'{ttl}, adults')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)); cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cb.set_label('Change in life satisfaction when 1 h moves from X to Y (points)', color=INK2, fontsize=8.5); cb.ax.tick_params(labelsize=8, colors=INK2)
save(fig, 'fig2_reallocation_matrices', bbox_inches='tight')

# ---------------------------------------------------------------- Fig. 3 gradient of decline
G = pd.read_csv(os.path.join(OUT_S, 'me_grid.csv')); g = G[G.domain == 'Eldercare'].sort_values('cagr')
# the tercile estimates are reported in Table S4(b) and are not drawn
T2e = T2[(T2.domain == 'Eldercare') & T2.column.isin(['Shrinking SMCs', 'Non-shrinking SMCs'])].set_index('column')
P = load_panel(); S = analytic_sample(P); smc = S[(S.scale == 'SMC') & (S.age_grp == '15-64')]; mun = smc.drop_duplicates('muni').cagr_17_23 * 100
thr_mean = {'Shrinking SMCs': mun[mun <= SHRINK_THRESHOLD * 100].mean(), 'Non-shrinking SMCs': mun[mun > SHRINK_THRESHOLD * 100].mean()}
fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor=SURF); ax.set_facecolor(SURF); ax.axhline(0, color=GREY2, lw=1)
xr = (g.cagr >= -3.5) & (g.cagr <= 1.0); ax.fill_between(g.cagr[xr], g.lo[xr], g.hi[xr], color=BLUE, alpha=0.12, lw=0); ax.plot(g.cagr[xr], g.est[xr], color=BLUE, lw=2)
ax.axvline(SHRINK_THRESHOLD * 100, color=GREY, lw=1, ls='--'); ax.text(SHRINK_THRESHOLD * 100 + 0.03, 0.62, 'threshold\n−0.15%/yr', fontsize=7.8, color=INK2, va='top', ha='left')
for key in ['Shrinking SMCs', 'Non-shrinking SMCs']:
    r = T2e.loc[key]; col = ORANGE if r.p < 0.05 else GREY2; xm = thr_mean[key]
    ax.errorbar(xm, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt='s', color=col, ms=7, capsize=3, lw=1.6, zorder=4)   # t(G-1) interval from the table
    if key == 'Shrinking SMCs': ax.text(xm, r.hi + 0.03, f'{key}\n{r.est:+.2f}{stars(r.p)}', fontsize=8, color=INK, va='bottom', ha='center')
    else: ax.text(xm + 0.08, r.est, f'{key}\n{r.est:+.2f}{stars(r.p)}', fontsize=8, color=INK, va='center')
ax.scatter(mun.clip(-3.5, 1.0), np.full(len(mun), -0.88), marker='|', s=40, color=GREY2, alpha=0.5, lw=0.8); ax.text(-3.45, -0.84, 'municipalities', fontsize=7.6, color=INK2, va='bottom')
ax.set_xlim(-3.6, 1.1); ax.set_ylim(-0.95, 0.85); ax.set_xlabel('Municipal population change 2017–2023 (%/yr; left = faster decline)', color=INK2, fontsize=8.5); ax.set_ylabel('Value of +1 h eldercare (points, 95% CI)', color=INK)
ax.set_title('Eldercare value along the gradient of decline, SMC adults', loc='left', fontsize=9.5, color=INK)
ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label='Continuous model (all SMC adults)'), Line2D([0], [0], marker='s', color=ORANGE, lw=0, ms=7, label='Threshold groups (Table 2)')], loc='upper left', fontsize=7.8, frameon=False)
fig.tight_layout(); save(fig, 'fig3_gradient')

# ---------------------------------------------------------------- Fig. S1 (supplement): moderator interactions as a coefficient plot
# the same domain x indicator interactions as Table S4, one panel per indicator, with 95% t(G-1) intervals, so that the
# reader sees the uncertainty of every cell rather than a colour. Merger indicator: Table S4 only.
MO = pd.read_csv(os.path.join(OUT_S, 'tableS4_moderators.csv')); MO = MO[MO.moderator != 'D_merged_z']; MODS = list(dict.fromkeys(zip(MO.moderator, MO.label)))
fig, axes = plt.subplots(1, len(MODS), figsize=(2.6 * len(MODS), 4.6), sharey=True, facecolor=SURF); y = np.arange(len(ORD))[::-1]
for ax, (m, lab) in zip(np.atleast_1d(axes), MODS):
    ax.set_facecolor(SURF); ax.axvline(0, color=GREY2, lw=1)
    for grp, off, mk, fc, base in [('SMC adults', +0.17, 'o', None, INK2), ('Large adults', -0.17, 's', SURF, GREY2)]:
        s_ = MO[(MO.group == grp) & (MO.moderator == m)].set_index('domain')
        for X, yi in zip(ORD, y):
            r = s_.loc[X]; col = ORANGE if r.p < 0.05 else base; lo, hi = r.est_oriented - r.crit * r.se, r.est_oriented + r.crit * r.se
            ax.hlines(yi + off, lo, hi, color=col, lw=1.6); ax.scatter(r.est_oriented, yi + off, s=30, marker=mk, color=col, facecolor=fc or col, zorder=3, linewidth=1.3)
    ax.set_title(lab.replace(' (−CAGR 2017–23)', '\n(−CAGR 2017–2023)'), loc='left', fontsize=9, color=INK)
    ax.set_xlim(-0.6, 0.6); ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_xticklabels(['−0.5', '', '0', '', '+0.5']); ax.tick_params(axis='x', labelsize=8)
np.atleast_1d(axes)[0].set_yticks(y); np.atleast_1d(axes)[0].set_yticklabels(ORD, color=INK)
fig.supxlabel('Change in the domain value per +1 SD of the indicator (0 → 1 for non-metropolitan), points per hour, 95% CI', color=INK2, fontsize=8.8, y=0.04)
fig.legend(handles=[Line2D([0], [0], marker='o', color=INK2, lw=1.6, label=f"SMC adults ({int(MO[MO.group == 'SMC adults'].N.iloc[0]):,} person-years)"), Line2D([0], [0], marker='s', color=GREY2, mfc=SURF, lw=1.6, label=f"Large-city adults ({int(MO[MO.group == 'Large adults'].N.iloc[0]):,})"),
                    Line2D([0], [0], marker='o', color=ORANGE, lw=1.6, label='p < 0.05')], loc='lower center', ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'figS1_moderator_grids', bbox_inches='tight')

# ---------------------------------------------------------------- Fig. 4 event study (dynamics only; the post-onset averages are Table 3 and Table S5)
ST = pd.read_csv(os.path.join(OUT_T, 'table3_eventstudy.csv')); DY = pd.read_csv(os.path.join(OUT_S, 'es_dynamic.csv'))
RELX = {'<=-3': -3, '-2': -2, '-1 (ref)': -1, '0': 0, '+1': 1, '>=+2': 2}; DY['x'] = DY.rel.map(RELX)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, facecolor=SURF)
YL = (np.floor(DY.lo.min() * 5) / 5 - 0.1, np.ceil(DY.hi.max() * 5) / 5 + 0.1)      # shared axis that contains every interval (no truncation)
for i, (ax, gname, ttl) in enumerate(zip(axes, GROUPS, TITLES)):
    ax.set_facecolor(SURF); d = DY[DY.group == gname].sort_values('x'); ons = int(ST[(ST.group == gname) & (ST.event == 'Eldercare onset')].onsets.iloc[0])
    ax.axhline(0, color=GREY2, lw=1); ax.axvline(-0.5, color=GREY, lw=1, ls='--')
    x = d.x.values; yv = d.est.values; lo = d.lo.values; hi = d.hi.values          # t(G-1) intervals from es_dynamic.csv
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.12, lw=0); ax.plot(x, yv, color=BLUE, lw=1.8, marker='o', ms=5)
    for xi, yi, p in zip(x, yv, d.p.values):
        if p < 0.05: ax.scatter(xi, yi, s=60, color=ORANGE, zorder=4, edgecolor=SURF, linewidth=1.2)
    ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_xticklabels(['≤−3', '−2', '−1', '0', '+1', '≥+2']); ax.set_ylim(*YL)
    ax.set_title(f'{ttl}, adults\n({ons} onsets)', loc='left', fontsize=9.5, color=INK); ax.set_xlabel('Years from eldercare onset (−1 = reference)', color=INK2, fontsize=8.5)
    if i == 0: ax.set_ylabel('Change in life satisfaction (points, 95% CI)', color=INK)
fig.legend(handles=[Line2D([0], [0], marker='o', color=BLUE, lw=1.8, label='estimate with 95% CI band'), Line2D([0], [0], marker='o', color=ORANGE, lw=0, label='p < 0.05')], loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'fig4_eventstudy')
