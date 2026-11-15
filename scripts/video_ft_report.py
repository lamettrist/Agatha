import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'models' / 'runs'
arms = sys.argv[1:] or ['shipped', 'control', 'video', 'video25']
out = []
P = out.append

# ---- sim gate
sim = {}
for a in arms:
    d = json.loads((R / f'video_ft_sim_{a}.json').read_text())
    row = {}
    for eng in ('new_projected', 'new_raw'):
        s = d['engines'][eng]['subsets']['live_physics']
        row[eng] = dict(chaotic=s['chaotic']['pos_err_m']['2s'], gentle=s['gentle']['pos_err_m']['2s'], pen=s['all']['penetration']['mean_m'] * 1e3,
                        drift=s['all']['rest_drift']['fd_speed_mps'], prec=s['all']['fell']['precision'], rec=s['all']['fell']['recall'],
                        ori=s['chaotic']['ori_err_deg']['2s'], n=s['all']['n_episodes'])
    sim[a] = row
P('### Sim ship gate (data/v2 test, live_physics, %d episodes)\n' % sim[arms[0]]['new_projected']['n'])
P('| Arm | Chaotic pos err @2s proj / raw (m) | Gentle pos err @2s proj / raw (m) | Penetration proj / raw (mm) | Rest drift proj (m/s) | Fall precision / recall (proj) | Chaotic ori err @2s proj (deg) |')
P('|---|---|---|---|---|---|---|')
for a in arms:
    p, r = sim[a]['new_projected'], sim[a]['new_raw']
    P(f"| {a} | {p['chaotic']:.4f} / {r['chaotic']:.4f} | {p['gentle']:.5f} / {r['gentle']:.5f} | {p['pen']:.3f} / {r['pen']:.3f} | {p['drift']:.4f} | {p['prec']:.3f} / {p['rec']:.3f} | {p['ori']:.1f} |")
P('')

cmp_ = json.loads((R / 'video_ft_compare.json').read_text())
def eng(ds, mode, a, sub='all_objects', kind='projected', group='datasets'):
    return cmp_[group][ds]['floor_modes'][mode]['engines'][f'{a}_{kind}' if a not in ('reference', 'hold_initial') else a][sub]
def pair(ds, mode, a, b, sub='all_objects', kind='projected', metric='rest_pos_err_m', group='datasets'):
    for row in cmp_[group][ds]['floor_modes'][mode]['paired']:
        if row['subset'] == sub and row['a'] == f'{a}_{kind}' and row['b'] == f'{b}_{kind}' and row['metric'] == metric: return row['result']
def ci(x): return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
def dl(x, scale=1, digits=3):
    if x is None: return '-'
    sig = ' *' if (x['ci95'][0] > 0 or x['ci95'][1] < 0) else ''
    return f"{x['delta']*scale:+.{digits}f} [{x['ci95'][0]*scale:+.{digits}f}, {x['ci95'][1]*scale:+.{digits}f}]{sig}"

for group, title in (('datasets', 'External ground truth: rest (final-frame) position error, m, mean [95% episode bootstrap CI], model + projection'),
                     ('video_derived', 'Held-out video-derived test split: rollout from the video-derived start state, error vs dataset ground truth at the last frame, m')):
    P(f'### {title}\n')
    P('| Dataset | Mode | Subset | n episodes / objects | hold still | ' + ' | '.join(arms) + ' |')
    P('|---|---|---|---|---|' + '---|' * len(arms))
    for ds, e in cmp_[group].items():
        for mode in ('slab', 'floor'):
            subs = ['all_objects'] if e.get('real_world') else ['all_objects', 'box_like_objects']
            for sub in subs:
                h = eng(ds, mode, 'hold_initial', sub, group=group)
                if not h['n_objects']: continue
                cells = [ci(eng(ds, mode, a, sub, group=group)['rest_pos_err_ci']) for a in arms]
                P(f"| {ds} | {mode} | {sub} | {h['n_episodes']} / {h['n_objects']} | {h['rest_pos_err_m']:.3f} | " + ' | '.join(cells) + ' |')
    P('')
    P('Paired differences in rest position error (B - A, m; negative = B better; 95% paired episode bootstrap; `*` = CI excludes 0):\n')
    prs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    P('| Dataset | Mode | Subset | ' + ' | '.join(f'{b} - {a}' for a, b in prs) + ' |')
    P('|---|---|---|' + '---|' * len(prs))
    for ds, e in cmp_[group].items():
        for mode in ('slab', 'floor'):
            subs = ['all_objects'] if e.get('real_world') else ['all_objects', 'box_like_objects']
            for sub in subs:
                if not eng(ds, mode, 'hold_initial', sub, group=group)['n_objects']: continue
                P(f'| {ds} | {mode} | {sub} | ' + ' | '.join(dl(pair(ds, mode, a, b, sub, group=group)) for a, b in prs) + ' |')
    P('')

P('### Real box-toss sets: more metrics (model + projection)\n')
P('| Dataset | Mode | Arm | pos err 0.5 s (m) | pos err 1 s (m) | rest pos err (m) | final ori err (deg) | rest face accuracy |')
P('|---|---|---|---|---|---|---|---|')
for ds, e in cmp_['datasets'].items():
    if not e.get('real_world'): continue
    for mode in ('slab', 'floor'):
        for a in ['hold_initial'] + arms:
            r = eng(ds, mode, a)
            P(f"| {ds} | {mode} | {a} | {r['pos_err_0.5s_m']:.3f} | {r['pos_err_1s_m']:.3f} | {r['rest_pos_err_m']:.3f} | {r['ori_err_final_deg']:.1f} | {r['rest_face']['accuracy']:.3f} |")
P('')

# ---- decision
P('### Decision rule, applied literally\n')
for v in [a for a in arms if a.startswith('video')]:
    P(f'**Arm `{v}`**\n')
    P('| Check | vs shipped | vs control |')
    P('|---|---|---|')
    ok_sim = True; cells = {}
    V = sim[v]['new_projected']
    rows = []
    for label, key, kind in (('sim chaotic pos err @2s (<= +2%)', 'chaotic', 'rel'), ('sim gentle pos err @2s (<= +2%)', 'gentle', 'rel'),
                             ('penetration (<= +0.02 mm)', 'pen', 'abs'), ('fall precision (>= -0.01)', 'prec', 'low'), ('fall recall (>= -0.01)', 'rec', 'low')):
        c = []
        for ref in ('shipped', 'control'):
            B = sim[ref]['new_projected']
            if kind == 'rel': d = V[key] / B[key] - 1; good = d <= .02; txt = f'{d*100:+.2f}%'
            elif kind == 'abs': d = V[key] - B[key]; good = d <= .02; txt = f'{d:+.4f} mm'
            else: d = V[key] - B[key]; good = d >= -.01; txt = f'{d:+.4f}'
            ok_sim &= good; c.append(f"{txt} {'PASS' if good else 'FAIL'}")
        P(f'| {label} | {c[0]} | {c[1]} |')
    ok_ext = True
    for ds in ('clevrer', 'kubric_movi_a_heldout'):
        for mode in ('slab', 'floor'):
            c = []
            for ref in ('shipped', 'control'):
                x = pair(ds, mode, ref, v)
                good = x['ci95'][1] < 0; c.append(f"{dl(x)} {'PASS' if good else 'FAIL'}")
                cells[(ds, mode, ref)] = good
            P(f'| {ds} rest pos err improves, {mode} mode, all objects (paired CI below 0) | {c[0]} | {c[1]} |')
    # external passes if, for each dataset, some floor mode shows improvement vs both references
    ext = {ds: any(cells[(ds, m, 'shipped')] and cells[(ds, m, 'control')] for m in ('slab', 'floor')) for ds in ('clevrer', 'kubric_movi_a_heldout')}
    ext_strict = {ds: all(cells[(ds, m, r)] for m in ('slab', 'floor') for r in ('shipped', 'control')) for ds in ext}
    P('')
    P(f"Sim conditions: {'all PASS' if ok_sim else 'FAIL'}. External improvement vs both references (in at least one floor mode): "
      + ', '.join(f"{ds} {'yes' if ok else 'no'}" for ds, ok in ext.items()) + '; in both floor modes: ' + ', '.join(f"{ds} {'yes' if ok else 'no'}" for ds, ok in ext_strict.items())
      + f". **Kept: {'YES' if ok_sim and all(ext.values()) else 'NO'}**\n")
print('\n'.join(out))
