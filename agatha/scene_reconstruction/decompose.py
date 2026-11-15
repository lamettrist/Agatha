"""Isolated native convex decomposition; a native failure cannot kill the API worker."""
import json,sys
from pathlib import Path
import numpy as np
import trimesh
import coacd

if __name__=='__main__':
    mesh=trimesh.load_mesh(sys.argv[1]);coacd.set_log_level('error')
    hulls=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32)),
        threshold=.12,max_convex_hull=8,resolution=800,mcts_nodes=10,mcts_iterations=20,
        preprocess_resolution=30,max_ch_vertex=128,seed=17)
    Path(sys.argv[2]).write_text(json.dumps({'parts':[{'type':'convex_hull','vertices':v.tolist()} for v,f in hulls]},allow_nan=False))
