"""Run the whole pipeline: sample -> reallocation model -> sensitivity -> event study -> figures.

Usage:
    python run_all.py --panel /path/to/panel.csv [--outdir output] [--steps s01,s02,s03,s04,s05]

--panel   person-year panel CSV (layout in data/schema.csv); default data/panel.csv
--outdir  where output/main, output/supplement and output/figures are written; default output/
--steps   comma-separated subset of the five steps (s05 needs the tables written by s01-s04)

The arguments are passed to the step scripts through the environment variables TSQ_PANEL and TSQ_OUTDIR, so each
script can also be run on its own from src/ (e.g. `cd src; python s05_figures.py`) with the same settings."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, 'src')
STEPS = {'s01': 's01_sample.py', 's02': 's02_reallocation.py', 's03': 's03_sensitivity.py', 's04': 's04_eventstudy.py', 's05': 's05_figures.py'}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--panel', default=os.path.join(HERE, 'data', 'panel.csv'), help='person-year panel CSV (see data/schema.csv)')
ap.add_argument('--outdir', default=os.path.join(HERE, 'output'), help='output directory (created if missing)')
ap.add_argument('--steps', default=','.join(STEPS), help='comma-separated subset of ' + ','.join(STEPS))
args = ap.parse_args()

if not os.path.exists(args.panel):
    sys.exit(f'panel CSV not found: {args.panel}\n(the individual-level data are not distributed with this repository; see README, Data availability)')
env = dict(os.environ, TSQ_PANEL=os.path.abspath(args.panel), TSQ_OUTDIR=os.path.abspath(args.outdir))
t_all = time.time()
for key in args.steps.split(','):
    script = STEPS[key.strip()]
    t = time.time(); print(f'\n=== {script} ===', flush=True)
    r = subprocess.run([sys.executable, os.path.join(SRC, script)], cwd=SRC, env=env)
    if r.returncode != 0: sys.exit(f'{script} failed')
    print(f'--- {script} done in {time.time() - t:.0f} s', flush=True)
print(f'\nAll outputs written to {os.path.abspath(args.outdir)} ({time.time() - t_all:.0f} s)')
