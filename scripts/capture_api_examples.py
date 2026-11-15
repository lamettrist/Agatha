"""Record real request/response excerpts for every API route, from a RUNNING server.

    .venv/bin/python scripts/capture_api_examples.py --base http://127.0.0.1:8017 [--upload-base http://127.0.0.1:8017]
        [--photo tests/images/acceptance.jpg] [--video tests/images/deskspace.mp4] [--out docs/api-examples.json]

--base         server used for everything except uploads (CPU is enough).  Start it with OPENAI_API_KEY unset/empty,
               so the assistant example is the honest no-key shape and nothing billable is called.
--upload-base  server running in .venv-reconstruction with a GPU; when omitted (or an upload fails to connect) the
               upload examples of the existing --out file are kept and labelled with their original capture date.

Nothing here is invented: every response is what the server answered.  Long arrays are excerpted (see `note`).
"""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
LONG = 24  # numeric arrays longer than this are excerpted

def shrink(value):
    """Long arrays -> {_excerpt, _length}; everything else unchanged."""
    if isinstance(value, dict): return {k: shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > LONG: return {'_excerpt': [shrink(v) for v in value[:6]], '_length': len(value)}
        return [shrink(v) for v in value]
    return value

def excerpt_run(result):
    """One engine result: keep first and last frame, record the real frame count."""
    if not result: return result
    result = dict(result); frames = result['frames']
    result['frame_count'] = len(frames); result['frames'] = [frames[0], frames[-1]]
    return result

def excerpt_simulation(body):
    if not isinstance(body, dict) or 'physics' not in body: return body
    body = dict(body); body['physics'] = excerpt_run(body['physics']); body['learned'] = excerpt_run(body['learned'])
    if body.get('error'):
        body['error'] = dict(body['error']); rows = body['error']['per_frame']
        body['error']['per_frame_count'] = len(rows); body['error']['per_frame'] = [rows[0], rows[-1]]
    return body

class Recorder:
    def __init__(self, base, timeout):
        self.base = base.rstrip('/'); self.client = httpx.Client(timeout=timeout)
    def call(self, method, path, json_body=None, params=None, files=None, data=None, describe=None, transform=None):
        start = time.perf_counter()
        response = self.client.request(method, self.base+path, json=json_body, params=params, files=files, data=data)
        elapsed = round(time.perf_counter()-start, 3)
        try:
            if 'json' not in response.headers.get('content-type', ''): raise ValueError('binary')
            body = response.json()
        except ValueError: body = {'_non_json_bytes': len(response.content), '_content_type': response.headers.get('content-type')}
        raw = body
        if transform: body = transform(body)
        entry = dict(method=method, path=path+('?'+'&'.join(f'{k}={v}' for k, v in params.items()) if params else ''))
        if json_body is not None: entry['request'] = json_body
        if describe is not None: entry['request_multipart'] = describe
        entry.update(status=response.status_code, elapsed_s=elapsed, response=shrink(body))
        return entry, raw

def capture(base, upload_base, photo, video, out):
    previous = json.loads(out.read_text()) if out.exists() else {}
    r = Recorder(base, 120); examples = {}
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    examples['note'] = ('Actual HTTP exchanges with a running server (uvicorn agatha.main:app) built from the working tree; nothing is invented. '
                        'Excerpting: simulation `frames` keep only the first and last frame (`frame_count` is the real count), '
                        '`error.per_frame` likewise (`per_frame_count`), and any array longer than 24 items is replaced by '
                        '{"_excerpt": first 6 items, "_length": real length}. `elapsed_s` is the wall time of that one request '
                        'seen by the client on the capture machine (first call of a model includes loading its checkpoint). '
                        'Scene IDs belong to the temporary capture server, not to a live deployment.')
    examples['captured'] = dict(utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), base=base,
                                upload_base=upload_base, git_head=commit+' + uncommitted working tree')
    examples['health'], _ = r.call('GET', '/api/health')
    scenes = {}
    for scenario in ('desk', 'bottle', 'laptop', 'soft'):
        examples['demo_'+scenario], raw = r.call('POST', '/api/scenes/demo', json_body={}, params=dict(scenario=scenario))
        scenes[scenario] = raw['scene_id']
    examples['demo_unknown_scenario_422'], _ = r.call('POST', '/api/scenes/demo', json_body={}, params=dict(scenario='kitchen'))
    examples['get_scene'], _ = r.call('GET', '/api/scenes/'+scenes['desk'])
    examples['get_scene_404'], _ = r.call('GET', '/api/scenes/'+'0'*32)

    def simulate(name, scenario, actions, **extra):
        request = {**dict(scene_id=scenes[scenario], actions=actions, engine='both', duration=3, fps=30), **extra}
        examples[name], raw = r.call('POST', '/api/simulate', json_body=request, transform=excerpt_simulation)
        examples[name]['scenario'] = scenario
        return raw
    push_mug = [dict(type='push', id='mug', direction=[1, 0], impulse_Ns=.55)]
    box = simulate('simulate_box_both', 'desk', push_mug)
    examples['leaderboard_post'], _ = r.call('POST', '/api/leaderboard', json_body=dict(name='Example team', scene_id=scenes['desk'], chaos_score=box['physics']['chaos_score']))
    examples['leaderboard_post_wrong_score_422'], _ = r.call('POST', '/api/leaderboard', json_body=dict(name='Example team', scene_id=scenes['desk'], chaos_score=box['physics']['chaos_score']+1))
    simulate('simulate_box_both_warm', 'desk', push_mug)
    simulate('simulate_box_all_actions', 'desk', [dict(type='set_mass', id='book1', kg=.8), dict(type='remove', id='phone'),
             dict(type='tilt_support', angle_deg=15, axis='z'), dict(type='set_gravity', g=3), dict(type='set_support_collisions', enabled=True),
             dict(type='push', id='mug', direction=[0, 1, 0], impulse_Ns=.3)])
    simulate('simulate_box_learned_only', 'desk', push_mug, engine='learned')
    simulate('simulate_box_physics_only', 'desk', push_mug, engine='physics')
    simulate('simulate_box_mlp_multistep', 'desk', push_mug, learned_architecture='mlp_multistep')
    simulate('simulate_joint_both_null_physics', 'laptop', [])
    simulate('simulate_joint_both_warm', 'laptop', push_mug)
    simulate('simulate_joint_physics_only_all_null', 'laptop', [], engine='physics')
    simulate('simulate_joint_forced_contact_gnn', 'laptop', [], learned_architecture='contact_gnn', engine='learned')
    simulate('simulate_soft_both_null_physics', 'soft', [])
    simulate('simulate_soft_both_warm', 'soft', [])
    simulate('simulate_422_joint_gnn_on_box_scene', 'desk', [], learned_architecture='joint_gnn')
    simulate('simulate_422_unknown_object', 'desk', [dict(type='push', id='nope', direction=[1, 0], impulse_Ns=.5)])
    simulate('simulate_422_validation', 'desk', [dict(type='push', id='mug', direction=[1, 0], impulse_Ns=99)])
    examples['simulate_404_unknown_scene'], _ = r.call('POST', '/api/simulate', json_body=dict(scene_id='0'*32, actions=[]))

    examples['metrics'], _ = r.call('GET', '/api/metrics')
    examples['leaderboard_post_before_physics_422'], _ = r.call('POST', '/api/leaderboard', json_body=dict(name='Example team', scene_id=scenes['bottle'], chaos_score=1.0))
    examples['leaderboard_get'], _ = r.call('GET', '/api/leaderboard')
    examples['assistant_no_key'], _ = r.call('POST', '/api/assistant/query', json_body=dict(scene_id=scenes['desk'], question='What if I push the mug?'))
    edit = '/api/scenes/{}/objects/{}'
    examples['object_edit_box_scene'], _ = r.call('POST', edit.format(scenes['desk'], 'mug'), json_body=dict(mass_kg=.4, shape='cylinder', size=[.09, .1, .09]))
    examples['object_edit_jointed_mass_only'], _ = r.call('POST', edit.format(scenes['laptop'], 'mug'), json_body=dict(mass_kg=.4))
    examples['object_edit_jointed_size_422'], _ = r.call('POST', edit.format(scenes['laptop'], 'laptop_lid'), json_body=dict(size=[.3, .02, .2]))
    examples['object_edit_soft_size_422'], _ = r.call('POST', edit.format(scenes['soft'], 'box_a'), json_body=dict(size=[.3, .02, .2]))
    examples['object_edit_unknown_object_404'], _ = r.call('POST', edit.format(scenes['desk'], 'nope'), json_body=dict(mass_kg=1))
    examples['reconstruction_asset_404'], _ = r.call('GET', '/api/reconstructions/'+'0'*64+'/overlay.jpg')
    examples['upload_no_file_422'], _ = r.call('POST', '/api/scenes', data=dict(table_width_m='1.2'))

    uploads = {}
    if upload_base:
        u = Recorder(upload_base, 600)
        try:
            uploads['health_upload_server'], _ = u.call('GET', '/api/health')
            if photo and photo.exists():
                for name in ('photo_upload', 'photo_upload_repeat'):
                    uploads[name], raw = u.call('POST', '/api/scenes', files={'image': (photo.name, photo.read_bytes(), 'image/jpeg')},
                                                data=dict(table_width_m='1.2'), describe=dict(image=str(photo), table_width_m='1.2'))
                prefix = raw.get('report', {}).get('debug_url') if isinstance(raw, dict) else None   # a URL PREFIX, not a page
                if prefix:
                    uploads['reconstruction_asset_png'], _ = u.call('GET', prefix+'/masks_overlay.png')
                    uploads['reconstruction_debug_url_bare_404'], _ = u.call('GET', prefix)
            if video and video.exists():
                for name in ('video_upload', 'video_upload_cached_repeat'):
                    uploads[name], _ = u.call('POST', '/api/scenes/video', files={'file': (video.name, video.read_bytes(), 'video/mp4')},
                                              describe=dict(file=str(video)))
        except httpx.HTTPError as exc:
            print(f'upload capture stopped: {type(exc).__name__}: {exc}')
    if not any(k.startswith('photo_upload') for k in uploads):
        old = previous.get('photo') or previous.get('photo_upload')
        if old:
            stamp = subprocess.run(['git', 'log', '-1', '--format=%cI', '--', 'docs/api-examples.json'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            uploads['photo_upload_previous_capture'] = dict(old, stale_note='NOT re-captured in this run; kept from the previous docs/api-examples.json '
                                                            f'(last committed {stamp}). The report shape may have changed since.')
    examples.update(uploads)
    out.write_text(json.dumps(examples, indent=1, allow_nan=False))
    print(f'Saved {len(examples)-2} recorded exchanges to {out}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='http://127.0.0.1:8017'); parser.add_argument('--upload-base', default=None)
    parser.add_argument('--photo', type=Path, default=ROOT/'tests/images/acceptance.jpg')
    parser.add_argument('--video', type=Path, default=ROOT/'tests/images/deskspace.mp4')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/api-examples.json')
    args = parser.parse_args()
    capture(args.base, args.upload_base, args.photo, args.video, args.out)
