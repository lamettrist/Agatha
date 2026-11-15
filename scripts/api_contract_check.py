"""Replay the HTTP calls an external frontend makes against several running servers and diff the answers.

Usage (the LAST server is the candidate; every earlier one is a baseline it is compared with):

    python scripts/api_contract_check.py master=http://127.0.0.1:8051 before=http://127.0.0.1:8052 \
        current=http://127.0.0.1:8053 [--upload-image tests/images/acceptance.jpg] [--numeric-baseline before]
        [--out-md docs/api-compat-report.md] [--out-json docs/api-compat-report.json]

For every call it records the status code, the recursive key/type structure and the values, then reports per
baseline: status changes, keys removed, keys added, type changes, value changes on shared keys and, for
/api/simulate, the largest absolute difference of every number in physics.frames / learned.frames.
Nothing is assumed about the servers: only what they answer is written down. Standard library only.
"""
import argparse
import json
import math
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request

ORIGIN = 'https://precognition.base44.app'
VOLATILE = {'scene_id', 'edited_from', 'run_id', 'elapsed_s', 'timings', 'seconds'}  # values never compared
TOL = 1e-9
# answers that depend on server history or on a non-repeatable reconstruction: structure is compared, values are not
STRUCTURE_ONLY = ('GET /api/leaderboard', 'POST /api/scenes (photo)')


def call(base, method, path, body=None, multipart=None, headers=None, timeout=600):
    data, hdr = None, dict(headers or {})
    if multipart is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in multipart.items():
            if isinstance(value, tuple):
                filename, content, ctype = value
                parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                              f'Content-Type: {ctype}\r\n\r\n').encode() + content + b'\r\n')
            else:
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        data = b''.join(parts) + f'--{boundary}--\r\n'.encode()
        hdr['Content-Type'] = 'multipart/form-data; boundary=' + boundary
    elif body is not None:
        data = json.dumps(body).encode()
        hdr['Content-Type'] = 'application/json'
    req = request.Request(base.rstrip('/') + path, data=data, method=method, headers=hdr)
    started = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as r:
            status, raw, rh = r.status, r.read(), dict(r.headers)
    except error.HTTPError as e:
        status, raw, rh = e.code, e.read(), dict(e.headers)
    except Exception as e:  # connection refused, timeout
        return dict(status=None, body=None, transport_error=repr(e), seconds=round(time.time() - started, 3))
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = {'_non_json': raw[:200].decode('latin1')}
    return dict(status=status, body=parsed, headers={k.lower(): v for k, v in rh.items()}, seconds=round(time.time() - started, 3))


# ---------------------------------------------------------------- the replayed calls
DESK_ACTIONS = {
    'push_mug': [dict(type='push', id='mug', direction=[1, 0], impulse_Ns=.55)],
    'remove_phone': [dict(type='remove', id='phone')],
    'set_mass_book1': [dict(type='set_mass', id='book1', kg=.8)],
    'tilt_support': [dict(type='tilt_support', angle_deg=15, axis='z')],
    'set_gravity': [dict(type='set_gravity', g=3)],
    'support_collisions_off': [dict(type='set_support_collisions', enabled=False)],
}
TOSS = [dict(type='push', id='bottle', direction=[0, 1, 0], impulse_Ns=1.5)]
ARCHS = [None, 'default', 'mlp_multistep', 'graph_impulse']
NEW_ARCHS = ['auto', 'contact_gnn', 'joint_gnn', 'soft_gnn']


def run_server(name, base, upload_image=None):
    """Returns {case_id: {group, new_feature, request, status, body, seconds}} for one server."""
    out = {}

    def rec(case, group, method, path, body=None, new_feature=False, **kw):
        r = call(base, method, path, body=body, **kw)
        r.update(group=group, new_feature=new_feature, request=dict(method=method, path=path, body=body))
        out[case] = r
        return r

    scenes = {}
    for case, path, key in (('demo/no_scenario', '/api/scenes/demo', 'desk'), ('demo/scenario=desk', '/api/scenes/demo?scenario=desk', None),
                            ('demo/scenario=bottle', '/api/scenes/demo?scenario=bottle', 'bottle')):
        r = rec(case, 'POST /api/scenes/demo', 'POST', path, body={})
        if key and r['status'] == 200: scenes[key] = r['body']['scene_id']
    for scenario in ('laptop', 'soft'):
        r = rec(f'demo/scenario={scenario}', 'POST /api/scenes/demo', 'POST', f'/api/scenes/demo?scenario={scenario}', body={}, new_feature=True)
        # Old servers ignore unknown scenarios and hand back the desk; only keep it when it is the real scene.
        if r['status'] == 200 and (r['body'].get('report') or {}).get('scenario') == scenario: scenes[scenario] = r['body']['scene_id']
    rec('demo/scenario=unknown', 'POST /api/scenes/demo', 'POST', '/api/scenes/demo?scenario=nonsense', body={})
    desk, bottle = scenes.get('desk'), scenes.get('bottle')
    rec('get_scene/desk', 'GET /api/scenes/{id}', 'GET', f'/api/scenes/{desk}')
    rec('get_scene/unknown_404', 'GET /api/scenes/{id}', 'GET', '/api/scenes/' + '0' * 32)
    rec('get_scene/malformed_id', 'GET /api/scenes/{id}', 'GET', '/api/scenes/not-a-scene')

    def sim(case, scene_id, actions, engine=None, arch=None, duration=None, fps=None, new_feature=False, group='POST /api/simulate'):
        body = dict(scene_id=scene_id, actions=actions)
        if engine: body['engine'] = engine
        if arch: body['learned_architecture'] = arch
        if duration is not None: body['duration'] = duration
        if fps is not None: body['fps'] = fps
        return rec(case, group, 'POST', '/api/simulate', body=body, new_feature=new_feature)

    sim('simulate/documented_example', desk, DESK_ACTIONS['push_mug'], 'both', None, 3, 30)
    sim('simulate/defaults_only', desk, DESK_ACTIONS['push_mug'])
    for engine in ('physics', 'learned', 'both'):
        for arch in ARCHS:
            for label, actions in DESK_ACTIONS.items():
                sim(f'simulate/desk/{label}/engine={engine}/arch={arch or "omitted"}/fps=30/dur=3', desk, actions, engine, arch, 3, 30)
            sim(f'simulate/bottle/toss_bottle/engine={engine}/arch={arch or "omitted"}/fps=30/dur=3', bottle, TOSS, engine, arch, 3, 30)
        for fps in (10, 20, 30):
            for duration in (.5, 3, 5):
                if (fps, duration) == (30, 3): continue
                sim(f'simulate/desk/push_mug/engine={engine}/arch=omitted/fps={fps}/dur={duration}', desk, DESK_ACTIONS['push_mug'], engine, None, duration, fps)
    sim('simulate/bottle/toss_no_support/engine=both', bottle, [dict(type='set_support_collisions', enabled=False)] + TOSS, 'both', None, 3, 30)
    # error cases
    sim('simulate/error/unknown_scene_404', '0' * 32, [], 'physics')
    sim('simulate/error/bad_action_type_422', desk, [dict(type='explode', id='mug')], 'physics')
    sim('simulate/error/bad_action_field_422', desk, [dict(type='push', id='mug', direction=[0, 0], impulse_Ns=.5)], 'physics')
    sim('simulate/error/unknown_object_422', desk, [dict(type='push', id='ghost', direction=[1, 0], impulse_Ns=.5)], 'both')
    sim('simulate/error/bad_fps_422', desk, [], 'physics', fps=24)
    sim('simulate/error/bad_duration_422', desk, [], 'physics', duration=9)
    sim('simulate/error/bad_engine_422', desk, [], 'bullet')
    sim('simulate/error/bad_architecture_422', desk, [], 'learned', 'transformer')
    rec('simulate/error/extra_field_422', 'POST /api/simulate', 'POST', '/api/simulate', body=dict(scene_id=desk, actions=[], colour='red'))
    # values/scenes that only newer servers know: a status change here is a new feature, not a regression
    for arch in NEW_ARCHS:
        sim(f'simulate/new/desk/push_mug/engine=both/arch={arch}', desk, DESK_ACTIONS['push_mug'], 'both', arch, 3, 30, new_feature=True)
    for scenario in ('laptop', 'soft'):
        if scenario in scenes:
            rec(f'get_scene/{scenario}', 'GET /api/scenes/{id}', 'GET', f'/api/scenes/{scenes[scenario]}', new_feature=True)
            for engine in ('physics', 'learned', 'both'):
                sim(f'simulate/new/{scenario}/no_action/engine={engine}', scenes[scenario], [], engine, None, 3, 30, new_feature=True)
            first = (out[f'demo/scenario={scenario}']['body']['scene']['objects'] or [{}])[-1].get('id')
            rec(f'object_edit/new/{scenario}/set_mass', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{scenes[scenario]}/objects/{first}', body=dict(mass_kg=.8), new_feature=True)
            rec(f'object_edit/new/{scenario}/set_size', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{scenes[scenario]}/objects/{first}', body=dict(size=[.1, .1, .1]), new_feature=True)
    # leaderboard
    rec('leaderboard/get_before', 'GET /api/leaderboard', 'GET', '/api/leaderboard')
    board_scene = (rec('leaderboard/_demo', '_setup', 'POST', '/api/scenes/demo', body={})['body'] or {}).get('scene_id')
    physics = sim('leaderboard/_simulate', board_scene, DESK_ACTIONS['push_mug'], 'physics', None, 3, 30, group='_setup')
    score = ((physics['body'] or {}).get('physics') or {}).get('chaos_score', 1.0)
    rec('leaderboard/post_latest_score', 'POST /api/leaderboard', 'POST', '/api/leaderboard', body=dict(name='Contract check', scene_id=board_scene, chaos_score=score))
    rec('leaderboard/post_wrong_score_422', 'POST /api/leaderboard', 'POST', '/api/leaderboard', body=dict(name='Contract check', scene_id=board_scene, chaos_score=score + 123))
    rec('leaderboard/post_unknown_scene_404', 'POST /api/leaderboard', 'POST', '/api/leaderboard', body=dict(name='x', scene_id='0' * 32, chaos_score=1))
    rec('leaderboard/post_missing_name_422', 'POST /api/leaderboard', 'POST', '/api/leaderboard', body=dict(scene_id=board_scene, chaos_score=score))
    rec('leaderboard/get_after', 'GET /api/leaderboard', 'GET', '/api/leaderboard')
    # assistant (servers are expected to run without API keys: documented no-key answer)
    rec('assistant/query', 'POST /api/assistant/query', 'POST', '/api/assistant/query', body=dict(scene_id=desk, question='What if I push the mug?'))
    rec('assistant/unknown_scene_404', 'POST /api/assistant/query', 'POST', '/api/assistant/query', body=dict(scene_id='0' * 32, question='hi'))
    rec('assistant/empty_question_422', 'POST /api/assistant/query', 'POST', '/api/assistant/query', body=dict(scene_id=desk, question=''))
    # endpoints missing from the README table
    rec('health/get', 'GET /api/health', 'GET', '/api/health')
    rec('metrics/get', 'GET /api/metrics', 'GET', '/api/metrics')
    rec('object_edit/set_mass', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{desk}/objects/mug', body=dict(mass_kg=.8))
    rec('object_edit/set_shape_size', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{desk}/objects/mug', body=dict(shape='cylinder', size=[.09, .1, .09]))
    rec('object_edit/unknown_object_404', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{desk}/objects/ghost', body=dict(mass_kg=.8))
    rec('object_edit/bad_shape_422', 'POST /api/scenes/{id}/objects/{object_id}', 'POST', f'/api/scenes/{desk}/objects/mug', body=dict(shape='sphere'))
    rec('reconstruction_asset/unknown_404', 'GET /api/reconstructions/{run_id}/{filename}', 'GET', '/api/reconstructions/' + 'a' * 64 + '/overlay.jpg')
    # uploads that need no vision model
    rec('upload/no_file_422', 'POST /api/scenes', 'POST', '/api/scenes', multipart={'table_width_m': '1.2'})
    rec('upload/not_an_image_422', 'POST /api/scenes', 'POST', '/api/scenes', multipart={'image': ('x.jpg', b'not an image', 'image/jpeg')})
    rec('upload_video/no_file_422', 'POST /api/scenes/video', 'POST', '/api/scenes/video', multipart={'table_width_m': '1.2'})
    rec('upload_scan/bad_suffix_422', 'POST /api/scenes/scan', 'POST', '/api/scenes/scan', multipart={'file': ('x.txt', b'hello', 'text/plain')})
    # CORS preflight from the Base44 origin (headers are compared as if they were a body)
    for path in ('/api/simulate', '/api/scenes'):
        r = call(base, 'OPTIONS', path, headers={'Origin': ORIGIN, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
        h = r.get('headers') or {}
        out[f'cors/preflight{path}'] = dict(group='OPTIONS (CORS preflight, Origin ' + ORIGIN + ')', new_feature=False, status=r['status'], seconds=r['seconds'],
                                           request=dict(method='OPTIONS', path=path, body=None),
                                           body={k: h.get(k) for k in ('access-control-allow-origin', 'access-control-allow-methods', 'access-control-allow-headers')})
    if upload_image:
        photo = Path(upload_image).read_bytes()
        rec('upload/photo', 'POST /api/scenes (photo)', 'POST', '/api/scenes', multipart={'image': (Path(upload_image).name, photo, 'image/jpeg'), 'table_width_m': '1.2'})
    return out


# ---------------------------------------------------------------- structure / value diff
def kind(v):
    if v is None: return 'null'
    if isinstance(v, bool): return 'bool'
    if isinstance(v, (int, float)): return 'number'
    if isinstance(v, str): return 'string'
    return 'array' if isinstance(v, list) else 'object'


def shape(v, path='$', acc=None):
    """{path: sorted set of JSON kinds}; array elements are merged under path[]."""
    acc = {} if acc is None else acc
    acc.setdefault(path, set()).add(kind(v))
    if isinstance(v, dict):
        for k, x in v.items(): shape(x, f'{path}.{k}', acc)
    elif isinstance(v, list):
        for x in v: shape(x, path + '[]', acc)
    return acc


def leaves(v, path='$', acc=None):
    acc = {} if acc is None else acc
    if isinstance(v, dict):
        for k, x in v.items():
            if k not in VOLATILE: leaves(x, f'{path}.{k}', acc)
    elif isinstance(v, list):
        acc[path + '.length'] = len(v)
        for i, x in enumerate(v): leaves(x, f'{path}[{i}]', acc)
    else: acc[path] = v
    return acc


def numbers(v, acc):
    if isinstance(v, bool): return acc
    if isinstance(v, (int, float)): acc.append(float(v))
    elif isinstance(v, dict):
        for k in sorted(v): numbers(v[k], acc)
    elif isinstance(v, list):
        for x in v: numbers(x, acc)
    return acc


def frames_diff(a, b, engine):
    """Largest |difference| over every number in <engine>.frames (objects matched by position in the list)."""
    fa, fb = ((a or {}).get(engine) or {}).get('frames'), ((b or {}).get(engine) or {}).get('frames')
    if fa is None and fb is None: return None
    if fa is None or fb is None: return dict(comparable=False, reason='frames present on one side only')
    na, nb = numbers(fa, []), numbers(fb, [])
    ids_a = [[o.get('id') for o in f.get('objects', [])] for f in fa]; ids_b = [[o.get('id') for o in f.get('objects', [])] for f in fb]
    if len(na) != len(nb) or ids_a != ids_b:
        return dict(comparable=False, reason=f'frame layout differs ({len(fa)} vs {len(fb)} frames, {len(na)} vs {len(nb)} numbers)')
    worst = max((abs(x - y) for x, y in zip(na, nb)), default=0.0)
    return dict(comparable=True, max_abs_diff=worst, numbers_compared=len(na), identical=worst == 0.0)


def strip_root(path, n=3):
    return path  # kept for readability of the JSON; the markdown generalises indices itself


def diff_case(old, new):
    d = dict(status_old=old['status'], status_new=new['status'], status_changed=old['status'] != new['status'],
             old_detail=old['body'].get('detail') if isinstance(old['body'], dict) and isinstance(old['body'].get('detail'), str) else None)
    so, sn = shape(old['body']), shape(new['body'])
    changed = sorted(p for p in so if p in sn and so[p] != sn[p])
    hidden = lambda p, roots: any(p.startswith(r + '.') or p.startswith(r + '[') for r in roots)
    # a subtree that is null/absent on one side shows up once (at its root), not once per descendant
    removed = sorted(p for p in so if p not in sn and not hidden(p, changed))
    removed = [p for p in removed if not hidden(p, [q for q in removed if q != p])]
    added = sorted(p for p in sn if p not in so and not hidden(p, changed))
    added = [p for p in added if not hidden(p, [q for q in added if q != p])]
    # "X[]" alone only means the array was empty on one side: a value difference, not a structural one
    removed, added = [p for p in removed if not p.endswith('[]')], [p for p in added if not p.endswith('[]')]
    d.update(keys_removed=removed, keys_added=added,
             type_changes=[dict(path=p, old=sorted(so[p]), new=sorted(sn[p])) for p in changed])
    lo, ln = (leaves(old['body']), leaves(new['body'])) if new['group'] not in STRUCTURE_ONLY else ({}, {})
    diffs = []
    for p in lo:
        if p not in ln: continue
        x, y = lo[p], ln[p]
        if kind(x) == 'number' and kind(y) == 'number':
            if abs(x - y) > TOL: diffs.append((p, x, y))
        elif x != y: diffs.append((p, x, y))
    d['value_changes_on_shared_keys'] = len(diffs)
    d['value_change_examples'] = [dict(path=p, old=x, new=y) for p, x, y in diffs[:6]]
    non_numeric = [(p, x, y) for p, x, y in diffs if not (kind(x) == 'number' and kind(y) == 'number') or p.endswith('.length')]
    d['non_numeric_value_changes'] = [dict(path=p, old=x, new=y) for p, x, y in non_numeric[:12]]
    if new['request']['path'] == '/api/simulate' and old['status'] == 200 and new['status'] == 200:
        d['physics_frames'] = frames_diff(old['body'], new['body'], 'physics')
        d['learned_frames'] = frames_diff(old['body'], new['body'], 'learned')
    return d


def classify(case, d, new_feature, numeric_must_match):
    """List of (severity, text). breaking / behaviour (numbers or texts differ on unchanged keys) / additive."""
    found = []
    if d['status_changed']:
        ok = lambda s: s is not None and 200 <= s < 300
        if not ok(d['status_old']) and ok(d['status_new']):
            # a request the old server refused is now served: no working client depended on the refusal
            found.append(('additive', f"status {d['status_old']} -> {d['status_new']} (request the baseline refused is now served: new endpoint, field or value)"))
        elif d['status_old'] in (404, 405) and not ok(d['status_new']) and d['old_detail'] in ('Not Found', 'Method Not Allowed'):
            found.append(('additive', f"status {d['status_old']} -> {d['status_new']} (route did not exist in the baseline)"))
        else: found.append(('breaking', f"status {d['status_old']} -> {d['status_new']}"))
        return found
    if new_feature and not (d['status_new'] is not None and 200 <= d['status_new'] < 300):
        if d['type_changes'] or d['value_changes_on_shared_keys']:
            found.append(('behaviour', 'error body differs for a request neither server serves (value unknown to the baseline, refused for another reason by the candidate)'))
        return found
    for p in d['keys_removed']: found.append(('breaking', f'key removed: {p}'))
    for t in d['type_changes']: found.append(('breaking', f"type change: {t['path']} {'|'.join(t['old'])} -> {'|'.join(t['new'])}"))
    for p in d['keys_added']: found.append(('additive', f'key added: {p}'))
    for engine in ('physics', 'learned'):
        f = d.get(engine + '_frames')
        if f and (not f['comparable'] or f['max_abs_diff'] > TOL):
            text = f"{engine}.frames numbers differ: " + (f"max abs diff {f['max_abs_diff']:.6g}" if f['comparable'] else f['reason'])
            found.append(('breaking' if numeric_must_match and not new_feature else 'behaviour', text))
    if d['value_changes_on_shared_keys'] and not any('frames numbers differ' in t for _, t in found):
        found.append(('behaviour', f"{d['value_changes_on_shared_keys']} value(s) differ on shared keys, e.g. {d['value_change_examples'][0]['path']}"))
    return found


# ---------------------------------------------------------------- report
def generalise(path):
    import re
    return re.sub(r'\[\d+\]', '[]', path)


def short(v, n=70):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n - 3] + '...'


def markdown(report):
    names = [s['name'] for s in report['servers']]
    cand = names[-1]
    L = ['# API backward-compatibility report', '',
         f"Generated by `scripts/api_contract_check.py` at {report['generated_utc']} UTC by replaying the same HTTP calls against running servers. "
         f"Candidate: **{cand}**. Everything below was observed in the answers; nothing is taken from the source.", '',
         '| Server | Base URL | Calls | Answered 2xx | 4xx | 5xx / no answer |', '| --- | --- | --- | --- | --- | --- |']
    for s in report['servers']:
        st = [c['status'] for c in report['raw_status'][s['name']].values()]
        L.append(f"| {s['name']} ({s.get('description') or '-'}) | {s['url']} | {len(st)} | {sum(1 for x in st if x and x < 300)} | {sum(1 for x in st if x and 400 <= x < 500)} | {sum(1 for x in st if not x or x >= 500)} |")
    L += ['', 'Classification: **breaking** = status change, removed key, type change (and, for the numeric baseline, different frame numbers for the same request); '
          '**behaviour** = same keys and types but different values; **additive** = new key, new endpoint or a newly accepted value.', '']
    for cmp in report['comparisons']:
        base = cmp['baseline']
        L += [f"## {base} -> {cand}", '']
        counts = cmp['summary']
        L += [f"Calls compared: {counts['cases']}. Calls with a breaking difference: **{counts['cases_breaking']}**; with a behaviour difference: {counts['cases_behaviour']}; "
              f"with additive differences only: {counts['cases_additive_only']}; identical: {counts['cases_identical']}.", '']
        for severity, title in (('breaking', 'Breaking'), ('behaviour', 'Behaviour (same shape, different values)'), ('additive', 'Additive')):
            rows = cmp['aggregated'][severity]
            L += [f'### {title}', '']
            if not rows: L += ['None observed.', '']; continue
            if severity == 'additive':  # one row per endpoint for added keys
                L += ['| Endpoint | Difference | Calls |', '| --- | --- | --- |']
                keys = {}
                for r in rows:
                    if r['text'].startswith('key added: '): keys.setdefault(r['group'], []).append(r['text'][len('key added: $.'):])
                    else: L.append(f"| `{r['group']}` | {r['text']}; e.g. `{r['example']}` | {r['count']} |")
                for group, added in keys.items(): L.append(f"| `{group}` | keys added: " + ', '.join(f'`{k}`' for k in added) + ' | |')
            else:
                L += ['| Endpoint | Difference | Calls | Example call |', '| --- | --- | --- | --- |']
                for r in rows: L.append(f"| `{r['group']}` | {r['text']} | {r['count']} | `{r['example']}` |")
            L.append('')
        n = cmp['numeric']
        L += ['### Frame numbers for identical /api/simulate requests', '',
              '| Engine | Calls with frames on both sides | Bit-identical | Largest abs difference | Call with the largest difference |', '| --- | --- | --- | --- | --- |']
        for engine in ('physics', 'learned'):
            e = n[engine]
            L.append(f"| {engine} | {e['compared']} | {e['identical']} | {e['max_abs_diff']:.6g} | `{e['worst_case'] or '-'}` |" if e['compared'] else f'| {engine} | 0 | - | - | - |')
        if n['by_architecture']:
            L += ['', '| learned_architecture sent | Calls | Learned frames identical | Largest abs difference |', '| --- | --- | --- | --- |']
            for arch, e in n['by_architecture'].items(): L.append(f"| {arch} | {e['compared']} | {e['identical']} | {e['max_abs_diff']:.6g} |")
        if cmp.get('added_key_samples'):
            L += ['', f'### Values of added keys as answered by {cand}', '', '| Call | Key | Value |', '| --- | --- | --- |']
            for e in cmp['added_key_samples']: L.append(f"| `{e['case']}` | `{e['path']}` | `{short(e['value'], 260)}` |")
        if cmp['value_examples']:
            L += ['', '### Examples of changed values', '', '| Call | Path | ' + base + ' | ' + cand + ' |', '| --- | --- | --- | --- |']
            for e in cmp['value_examples']: L.append(f"| `{e['case']}` | `{e['path']}` | `{short(e['old'])}` | `{short(e['new'])}` |")
        L.append('')
    if report.get('determinism'):
        d = report['determinism']
        L += ['## Repeatability control', '', f"The simulate calls were sent a second time to **{cand}** (fresh demo scenes). Physics frames: {d['physics']['identical']}/{d['physics']['compared']} bit-identical "
              f"(largest difference {d['physics']['max_abs_diff']:.6g}); learned frames: {d['learned']['identical']}/{d['learned']['compared']} bit-identical (largest difference {d['learned']['max_abs_diff']:.6g}).", '']
    L += ['## Status code of every call', '', '| Call | ' + ' | '.join(names) + ' |', '| --- |' + ' --- |' * len(names)]
    collapsed = {}
    for case in report['case_order']:
        row = tuple(report['raw_status'][n].get(case, {}).get('status', 'not sent') for n in names)
        key = generalise_case(case)
        collapsed.setdefault((key, row), []).append(case)
    for (key, row), cases in collapsed.items():
        L.append(f"| `{key}`{' (' + str(len(cases)) + ' calls)' if len(cases) > 1 else ''} | " + ' | '.join(str(x) for x in row) + ' |')
    L += ['', report.get('notes', ''), '']
    return '\n'.join(L)


def generalise_case(case):
    import re
    if case.startswith('simulate/desk/') or case.startswith('simulate/bottle/toss_bottle'):
        return re.sub(r'/arch=[^/]+/fps=\d+/dur=[\d.]+', '/arch=*/fps=*/dur=*', re.sub(r'simulate/desk/[^/]+', 'simulate/desk/<action>', case))
    return case


def numeric_summary(cases):
    out = {e: dict(compared=0, identical=0, max_abs_diff=0.0, worst_case=None, not_comparable=[]) for e in ('physics', 'learned')}
    by_arch = {}
    for case, d in cases.items():
        for engine in ('physics', 'learned'):
            f = d.get(engine + '_frames')
            if not f: continue
            if not f['comparable']: out[engine]['not_comparable'].append(case); continue
            o = out[engine]; o['compared'] += 1; o['identical'] += f['identical']
            if f['max_abs_diff'] > o['max_abs_diff']: o['max_abs_diff'], o['worst_case'] = f['max_abs_diff'], case
            if engine == 'learned':
                arch = (d['request_body'] or {}).get('learned_architecture', '(omitted)')
                a = by_arch.setdefault(arch, dict(compared=0, identical=0, max_abs_diff=0.0))
                a['compared'] += 1; a['identical'] += f['identical']; a['max_abs_diff'] = max(a['max_abs_diff'], f['max_abs_diff'])
    out['by_architecture'] = by_arch
    return out


def compare(old, new, numeric_must_match):
    cases, aggregated, examples = {}, {'breaking': {}, 'behaviour': {}, 'additive': {}}, []
    summary = dict(cases=0, cases_breaking=0, cases_behaviour=0, cases_additive_only=0, cases_identical=0)
    for case, n in new.items():
        if n['group'] == '_setup' or case not in old: continue
        o = old[case]
        d = diff_case(o, n); d['request_body'] = n['request']['body']
        found = classify(case, d, n['new_feature'], numeric_must_match)
        d['classification'] = [dict(severity=s, text=t) for s, t in found]
        cases[case] = d; summary['cases'] += 1
        severities = {s for s, _ in found}
        summary['cases_breaking' if 'breaking' in severities else 'cases_behaviour' if 'behaviour' in severities else 'cases_additive_only' if severities else 'cases_identical'] += 1
        for s, t in found:
            import re
            text = re.sub(r'max abs diff [\d.e+-]+', 'max abs diff (see table)', re.sub(r'^\d+ value\(s\) differ on shared keys, e\.g\. ', 'values differ, e.g. ', generalise(t)))
            row = aggregated[s].setdefault((n['group'], text), dict(group=n['group'], text=text, count=0, example=case))
            row['count'] += 1
        if d['non_numeric_value_changes'] and len([e for e in examples if e['case'].split('/')[0] == case.split('/')[0]]) < 4:
            seen = {(e['path'], short(e['old']), short(e['new'])) for e in examples}
            for e in d['non_numeric_value_changes'][:3]:
                key = (generalise(e['path']), short(e['old']), short(e['new']))
                if key not in seen: examples.append(dict(case=case, path=generalise(e['path']), old=e['old'], new=e['new'])); seen.add(key)
    return dict(summary=summary, aggregated={s: sorted(v.values(), key=lambda r: (r['group'], r['text'])) for s, v in aggregated.items()},
                numeric=numeric_summary(cases), value_examples=examples[:24], cases=cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('servers', nargs='*', help='name=base_url; the last one is the candidate')
    ap.add_argument('--describe', action='append', default=[], help='name=free text shown in the report')
    ap.add_argument('--numeric-baseline', action='append', default=[], help='baseline whose frame numbers must equal the candidate (differences are breaking)')
    ap.add_argument('--upload-image', help='also POST this photo to /api/scenes on every server (needs the vision models)')
    ap.add_argument('--upload-server', action='append', default=[], help='name=base_url of a vision-enabled instance of that server to send the photo to instead')
    ap.add_argument('--no-determinism', action='store_true', help='skip the second pass against the candidate')
    ap.add_argument('--notes', default='', help='free text appended to the markdown')
    ap.add_argument('--out-md', default='docs/api-compat-report.md')
    ap.add_argument('--out-json', default='docs/api-compat-report.json')
    ap.add_argument('--render-only', action='store_true', help='rebuild the markdown from an existing --out-json without sending any request')
    args = ap.parse_args()
    if args.render_only:
        Path(args.out_md).write_text(markdown(json.loads(Path(args.out_json).read_text())))
        return 0
    if len(args.servers) < 2: ap.error('give at least two name=base_url servers')
    servers = [dict(name=s.split('=', 1)[0], url=s.split('=', 1)[1]) for s in args.servers]
    descriptions = dict(x.split('=', 1) for x in args.describe)
    for s in servers: s['description'] = descriptions.get(s['name'])
    with ThreadPoolExecutor(len(servers)) as pool:
        results = dict(zip([s['name'] for s in servers], pool.map(lambda s: run_server(s['name'], s['url'], None if args.upload_server else args.upload_image), servers)))
    candidate = servers[-1]['name']
    if args.upload_image and args.upload_server:  # same code, separate GPU instances: only the photo goes there
        gpu = dict(x.split('=', 1) for x in args.upload_server)
        photo = Path(args.upload_image).read_bytes()
        def upload(name):
            r = call(gpu[name], 'POST', '/api/scenes', multipart={'image': (Path(args.upload_image).name, photo, 'image/jpeg'), 'table_width_m': '1.2'}, timeout=900)
            r.update(group='POST /api/scenes (photo)', new_feature=False, request=dict(method='POST', path='/api/scenes', body=None))
            return r
        with ThreadPoolExecutor(len(gpu)) as pool:
            for name, r in zip(gpu, pool.map(upload, gpu)): results[name]['upload/photo'] = r
    report = dict(generated_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), servers=servers, candidate=candidate, tolerance=TOL,
                  case_order=[c for c in results[candidate] if results[candidate][c]['group'] != '_setup'],
                  raw_status={n: {c: dict(status=r['status'], seconds=r['seconds']) for c, r in res.items() if r['group'] != '_setup'} for n, res in results.items()},
                  requests={c: r['request'] for c, r in results[candidate].items() if r['group'] != '_setup'},
                  comparisons=[], notes=args.notes)
    for s in servers[:-1]:
        cmp = compare(results[s['name']], results[candidate], s['name'] in args.numeric_baseline)
        cmp['baseline'] = s['name']
        report['comparisons'].append(cmp)
    if not args.no_determinism:
        again = run_server(candidate, servers[-1]['url'])
        report['determinism'] = numeric_summary({c: dict(diff_case(results[candidate][c], r), request_body=r['request']['body'])
                                                 for c, r in again.items() if c.startswith('simulate/') and c in results[candidate]})
    # one full answer per endpoint group and server, so the report can be checked against real payloads without re-running
    samples = {}
    for name, res in results.items():
        for case in ('demo/no_scenario', 'simulate/defaults_only', 'assistant/query', 'health/get', 'leaderboard/post_latest_score', 'leaderboard/get_after',
                     'object_edit/set_mass', 'demo/scenario=laptop', 'demo/scenario=soft', 'simulate/new/laptop/no_action/engine=both',
                     'simulate/new/soft/no_action/engine=both', 'upload/photo', 'cors/preflight/api/simulate', 'simulate/error/unknown_object_422',
                     'simulate/error/bad_action_type_422', 'simulate/error/unknown_scene_404'):
            if case in res:
                body = json.loads(json.dumps(res[case]['body']))
                for engine in ('physics', 'learned'):
                    r = body.get(engine) if isinstance(body, dict) else None
                    if r and len(r.get('frames', [])) > 2: r['frame_count'] = len(r['frames']); r['frames'] = [r['frames'][0], r['frames'][-1]]
                    for mesh in ((r or {}).get('soft_meshes') or {}).values():
                        for k, v in list(mesh.items()):
                            if isinstance(v, list) and len(v) > 4: mesh[k] = v[:2] + [f'... {len(v) - 2} more']
                if isinstance(body, dict) and isinstance(body.get('error'), dict) and len(body['error'].get('per_frame', [])) > 2:
                    body['error']['per_frame'] = [body['error']['per_frame'][0], body['error']['per_frame'][-1]]
                samples.setdefault(case, {})[name] = dict(status=res[case]['status'], body=body)
    report['samples_excerpted'] = samples
    def resolve(body, path):
        for part in path[2:].split('.'):
            body = body[part]
        return body
    for cmp in report['comparisons']:
        rows, seen = [], set()
        for case in samples:
            d = cmp['cases'].get(case)
            if not d or candidate not in samples[case] or d['status_changed']: continue
            for path in d['keys_added']:
                if '[]' in path or path in seen or path.startswith(('$.scene.objects', '$.report.stages', '$.report.metrics')): continue
                try: rows.append(dict(case=case, path=path, value=resolve(samples[case][candidate]['body'], path))); seen.add(path)
                except (KeyError, TypeError): pass
        cmp['added_key_samples'] = rows[:40]
    Path(args.out_json).write_text(json.dumps(report, indent=1, allow_nan=False, default=str))
    Path(args.out_md).write_text(markdown(report))
    for cmp in report['comparisons']:
        print(cmp['baseline'], '->', candidate, cmp['summary'])
    print('wrote', args.out_md, args.out_json)
    return 1 if any(c['summary']['cases_breaking'] for c in report['comparisons']) else 0


if __name__ == '__main__':
    sys.exit(main())
