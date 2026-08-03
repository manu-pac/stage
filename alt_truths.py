#dado o set e o nome da pasta dele, criar um arquivo com varias listas 
# (uma com a verdade de cada exemplo em cada mundo) - usando as funcoes 
# de check e as funcoes de criaçao p reconstruir
import argparse
from pathlib import Path
import classes as cl
import tf_generation as tfg
import pickle
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder",required=True, help="name of the dataset folder")

    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    folder = project_root / "dataset" / args.folder

    act_world = pickle.load(open(folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(folder / "alt_worlds.pkl", "rb"))

    dev_t = pickle.load(open(folder / "dev_t.pkl", "rb"))
    dev_f = pickle.load(open(folder / "dev_f.pkl", "rb"))

    number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = pickle.load(
        open(folder / "params.pkl", "rb")
    )
    tfg.setup(number_pl_=number_pl, max_depth_=max_depth, act_world_=act_world, alt_worlds_=alt_worlds)

    w_truths = {"dev_t":{},"dev_f":{}}

    for w in range(n_worlds-1):
        world = alt_worlds[w]
        i_func = cl.InterpretationFunc(set(world))
        w_truths["dev_t"][world] = []
        w_truths["dev_f"][world] = []

        for i in dev_t:
            f = tfg.true_le(i)
            truth = f.check(i_func)
            w_truths["dev_t"][world].append(truth)
        for i in dev_f:
            f = tfg.false_le(i)
            truth = f.check(i_func)
            w_truths["dev_f"][world].append(truth)

 
    mean_path = project_root/ "dataset" / args.folder / "alt_truths.pkl"
    with open(mean_path, "wb") as f:
        pickle.dump(w_truths, f)
    print(f"Saved alternative world truths to dataset/{args.folder}/alt_truths.pkl")

if __name__ == "__main__":
    main()