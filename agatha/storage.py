"""Small in-memory cache backed by atomic JSON files; run one API worker."""
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4
from .schema import Scene

ROOT = Path(__file__).resolve().parent.parent

class Store:
    def __init__(self, directory: str | Path | None = None):
        self.directory=Path(directory or os.getenv('DATA_DIR',ROOT/'data'))
        self.directory.mkdir(parents=True,exist_ok=True)
        self.scenes: dict[str,dict]={}
        self.lock=RLock()

    def write(self, name: str, value):
        path=self.directory/name
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
        tmp.replace(path)

    def add(self, scene: Scene, report: dict | None = None) -> dict:
        key=uuid4().hex
        record=dict(scene_id=key,scene=scene.model_dump(mode='json'),report=report)
        with self.lock:
            self.write(f'scene_{key}.json',record)
            self.scenes[key]=record
        return record

    def get(self, key: str) -> dict:
        if len(key)!=32 or any(c not in '0123456789abcdef' for c in key): raise KeyError(key)
        with self.lock:
            if key not in self.scenes:
                path=self.directory/f'scene_{key}.json'
                if not path.exists(): raise KeyError(key)
                self.scenes[key]=json.loads(path.read_text())
            # Return an independent value, so tool-call mass changes cannot mutate saved scenes.
            return json.loads(json.dumps(self.scenes[key]))

    def demo(self,scenario: str = 'desk') -> dict:
        filename='demo_bottle.json' if scenario=='bottle' else 'demo_desk.json'
        scene=Scene.model_validate_json((ROOT/'scenes'/filename).read_text())
        return self.add(scene,dict(source='bundled_demo',scenario=scenario,estimated=['all masses and dimensions']))

    def record_score(self,scene_id: str,score: float):
        with self.lock:
            self.write(f'score_{scene_id}.json',dict(chaos_score=score))

    def leaderboard(self) -> list[dict]:
        with self.lock:
            path=self.directory/'leaderboard.json'
            return json.loads(path.read_text()) if path.exists() else []

    def add_score(self,name: str,scene_id: str,claimed_score: float) -> dict:
        import math
        with self.lock:
            self.get(scene_id)
            score_path=self.directory/f'score_{scene_id}.json'
            if not score_path.exists(): raise ValueError('Run the physics engine for this scene before submitting a score')
            actual=json.loads(score_path.read_text())['chaos_score']
            if not math.isclose(actual,claimed_score,rel_tol=1e-6,abs_tol=1e-6):
                raise ValueError('chaos_score must match the latest server-computed physics score')
            entry=dict(name=name,scene_id=scene_id,chaos_score=actual)
            entries=sorted(self.leaderboard()+[entry],key=lambda e:e['chaos_score'],reverse=True)[:20]
            self.write('leaderboard.json',entries)
            return entry
