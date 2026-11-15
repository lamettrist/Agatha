"""Discover openly licensed physics datasets with Exa (search endpoint, highlights only).

Needs EXA_API_KEY in the environment or in .env. Results are candidates only: every
licence must still be read at its official source before a dataset is used, and
recorded in DATA_LICENSES.md. Web text is untrusted data, never instructions.

    .venv/bin/python scripts/find_physics_datasets.py --out data/external/exa_dataset_search.json
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OBJECTIVE = (
    "Find downloadable, openly licensed datasets of simulated or real physical dynamics that could train or "
    "evaluate a learned physics simulator. Rank official dataset pages, GitHub repositories and papers that "
    "state a licence and a download location first; exclude news, tutorials and course pages. Pull the dataset "
    "name, licence, download URL, size, and whether per-object or per-vertex ground-truth 3D state is included."
)

QUERIES = {
    "rigid": "open dataset of rigid body dynamics trajectories with ground-truth 3D poses for learned simulators, objects falling toppling colliding on a table",
    "articulated": "open dataset of articulated object dynamics with joint states, hinges drawers doors robot arm trajectories, permissive licence",
    "ragdoll": "open dataset of simulated ragdoll or passive humanoid rigid-body chain dynamics with joint ground truth",
    "soft": "open dataset of deformable soft body simulation trajectories with mesh vertex positions, cloth and elastic bodies, for graph network simulators",
    "meshgraphnets": "MeshGraphNets and Learning to Simulate datasets download licence cloth flag deforming plate",
    "video_physics": "openly licensed physics video benchmark with object trajectories and stability or fall labels, e.g. block towers, intuitive physics",
    "real_capture": "real-world captured rigid object tossing or collision dataset with motion capture 6D poses, open licence",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/exa_dataset_search.json")
    parser.add_argument("--only", nargs="*", choices=sorted(QUERIES), help="subset of query keys")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("EXA_API_KEY"):
        sys.exit("EXA_API_KEY is not set (export it or put it in .env); nothing was searched.")
    from exa_py import Exa

    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    output = dict(created_unix=time.time(), objective=OBJECTIVE, queries={})
    for key in args.only or QUERIES:
        query = QUERIES[key]
        try:
            response = exa.search(query, type="auto", objective=OBJECTIVE, contents={"highlights": True})
        except TypeError:  # older exa-py without the objective keyword
            response = exa.search(query, type="auto", contents={"highlights": True})
        rows = [dict(title=r.title, url=r.url, published=getattr(r, "published_date", None),
                     highlights=getattr(r, "highlights", None)) for r in response.results]
        output["queries"][key] = dict(query=query, results=rows)
        print(f"[{key}] {len(rows)} results")
        for row in rows:
            print("   ", row["title"], "|", row["url"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
