from classes import PLetter, Neg, Conj, InterpretationFunc, check_type
import string
import random
from pathlib import Path
import argparse
import pickle

number_pl = None
min_depth = None
max_depth = None
corpus_size = None
prop_tf = None
n_worlds = None

act_world = set()
alt_worlds = set()
letters = []
act_list = [] # sorted list of act_world letters, for indexing purposes
nact_list = [] # sorted list of letters NOT in act_world
Tc,Fc,Ts,Fs = {}, {}, {}, {}
true_cache, false_cache = {}, {}

def valuation(number_pl,prop_tf):
  # this function randomly assigns T/F to the desired number of propositional letters (number_pl) following the desired proportion of T/F (prop_t),
  # and returns the set of all true atomic formulas and the set of all false atomic formulas
  true = random.sample(letters,k=int(number_pl*prop_tf))
  return set(true), set(letters)-set(true)

def true(i,d):
  if (i,d) in true_cache:
    return true_cache[(i,d)]
  if d == 1:
    phi = PLetter(act_list[i]) # depth 1, can only be prop. letters

  elif i < Fc[d-1]: # inside negation block
      phi = Neg(false(i, d-1))

  else: # inside conjunction block
    j = i - Fc[d-1] #update index to align it with first conjunction group
    sizeA = Tc[d-1] * Ts[d-1]

    if j < sizeA: # first conjunction group = T^T, deep^anything
      row, column = divmod(j, Ts[d-1])
      phi = Conj(true(row, d-1), true_le(column))

    else: # second conjunction group = T^T, shallow^deep
      j -= sizeA
      row, column = divmod(j,Tc[d-1])
      phi = Conj(true_le(row), true(column, d-1))

  true_cache[(i,d)] = phi
  return phi

def true_le(i):
  e = 1
  while i >= Tc[e]:
    i -= Tc[e]
    e+=1
  return true(i,e)

def false(i,d):
  if (i,d) in false_cache:
    return false_cache[(i,d)]

  if d == 1:
    phi = PLetter(nact_list[i])

  elif i < Tc[d-1]: # negation block
    phi = Neg(true(i,d-1))

  else: #conjunctions block
    j = i - Tc[d-1]

    # case 1: F^F, deep^anything
    sizeA = Fc[d-1]*Fs[d-1]
    if j < sizeA:
      row, column = divmod(j, Fs[d-1])
      phi = Conj(false(row, d-1), false_le(column))
    else:
      j -= sizeA

      # case 2: F^F, shallow^deep
      sizeB = Fs[d-2]*Fc[d-1]
      if j < sizeB:
        row, column = divmod(j, Fc[d-1])
        phi = Conj(false_le(row), false(column, d-1))
      else:
        j -= sizeB

        # case 3: F^T, deep^anything
        sizeC = Fc[d-1]*Ts[d-1]
        if j < sizeC:
          row, column = divmod(j, Ts[d-1])
          phi = Conj(false(row, d-1), true_le(column))
        else:
          j -= sizeC

          # case 4: F^T, shallow^deep
          sizeD = Fs[d-2]*Tc[d-1]
          if j < sizeD:
            row, column = divmod(j, Tc[d-1])
            phi = Conj(false_le(row), true(column, d-1))
          else:
            j -= sizeD

            # case 5: T^F, deep^anything
            sizeE = Tc[d-1]*Fs[d-1]
            if j < sizeE:
              row, column = divmod(j, Fs[d-1])
              phi = Conj(true(row, d-1), false_le(column))
            else:
              j -= sizeE

              # case 6: T^F, shallow^deep
              sizeF = Ts[d-2]*Fc[d-1]
              if j < sizeF:
                row, column = divmod(j, Fc[d-1])
                phi = Conj(true_le(row), false(column, d-1))

  false_cache[(i,d)] = phi
  return phi

def false_le(i):
  e = 1
  while i >= Fc[e]:
    i -= Fc[e]
    e+=1
  return false(i,e)


# so i can use it elsewhere:

def build_index_tables(max_depth_):
  # (re)builds Tc, Fc, Ts, Fs up to max_depth_, based on the current act_world/number_pl.
  # Also clears the true()/false() caches, since they're indexed by (i,d) and would
  # otherwise mix formulas from different act_worlds together.
  global Tc, Fc, Ts, Fs, true_cache, false_cache, max_depth
 
  max_depth = max_depth_
 
  Tc = {1: len(act_world)}
  Fc = {1: number_pl - len(act_world)}
  Ts = {0: 0, 1: Tc[1]}
  Fs = {0: 0, 1: Fc[1]}
 
  for d in range(2, max_depth + 1):
    Tc[d] = Fc[d-1] + Tc[d - 1] * Ts[d - 1] + Ts[d - 2] * Tc[d - 1]
    Ts[d] = Ts[d - 1] + Tc[d]
 
    Fc[d] = Tc[d-1] + Fc[d-1] * Fs[d-1] + Fs[d-2] * Fc[d-1] + Fc[d-1] * Ts[d-1] + Fs[d-2] * Tc[d-1] + Tc[d-1] * Fs[d-1] + Ts[d-2] * Fc[d-1]
    Fs[d] = Fs[d - 1] + Fc[d]
 
  true_cache = {}
  false_cache = {}

def setup(number_pl_, max_depth_, act_world_, alt_worlds_):
  # this is the function that should be called from another script
  global number_pl, letters, act_world, alt_worlds, n_worlds, act_list, nact_list

  number_pl = number_pl_
  letters = list(string.ascii_lowercase)[:number_pl]
  act_world = set(act_world_)
  alt_worlds = set(alt_worlds_)
  act_list = sorted(act_world)
  nact_list = sorted(set(letters)-act_world)
  n_worlds = len(alt_worlds) + 1
 
  build_index_tables(max_depth_)

  
def prob(i):
    f = true_le(i)
    count = sum(f.check(InterpretationFunc(set(w))) for w in alt_worlds)
    is_suspect = (count == n_worlds - 1)
    return count/len(alt_worlds), is_suspect

def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_pl", type=int, required=True, help="Number of propositional letters")
    parser.add_argument("--min_depth", type=int, required=True, help="Minimum depth of formulas")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth of formulas")
    parser.add_argument("--corpus_size", type=int, required=True, help="Size of the training corpus")
    parser.add_argument("--prop_tf", type=float, default=0.5, help="Proportion of true/false values for propositional letters")
    parser.add_argument("--n_worlds", type=int, required=True, help="Number of alternative worlds")

    parser.add_argument("--folder_name", type=str, required=True, help="Name of the folder to save the dataset")

    args = parser.parse_args()

    global number_pl, min_depth, max_depth, corpus_size, prop_tf, n_worlds, letters, Tc, Fc, Ts, Fs, true_cache, false_cache, act_world, alt_worlds, act_list, nact_list

    number_pl = args.number_pl
    min_depth = args.min_depth
    max_depth = args.max_depth
    corpus_size = args.corpus_size
    prop_tf = args.prop_tf
    n_worlds = args.n_worlds

    letters = list(string.ascii_lowercase)[:number_pl]
    print(letters)

    alt_worlds = set()

    while len(alt_worlds) < n_worlds:
        t, f = valuation(number_pl, prop_tf)
        alt_worlds.add(frozenset(t))

    # actual world:
    act_world = set(alt_worlds.pop())
    act_list = sorted(act_world)
    nact_list = sorted(set(letters)-act_world)

    # initialize index counts
    Tc = {1: len(act_world)}
    Fc = {1: number_pl - len(act_world)}
    Ts = {0: 0, 1: Tc[1]}
    Fs = {0: 0, 1: Fc[1]}

    # progress index counts up to max_depth
    for d in range(2, max_depth + 1):
        Tc[d] = Fc[d-1] + Tc[d - 1] * Ts[d - 1] + Ts[d - 2] * Tc[d - 1]
        Ts[d] = Ts[d - 1] + Tc[d]

        Fc[d] = Tc[d-1] + Fc[d-1] * Fs[d-1] + Fs[d-2] * Fc[d-1] + Fc[d-1] * Ts[d-1] + Fs[d-2] * Tc[d-1] + Tc[d-1] * Fs[d-1] + Ts[d-2] * Fc[d-1]
        Fs[d] = Fs[d - 1] + Fc[d]

    true_cache = {}
    false_cache = {}
    
    samples_t = random.sample(range(Ts[min_depth-1], Ts[max_depth]), corpus_size*2)

    idx_t = samples_t[:corpus_size]
    dev_t = samples_t[corpus_size:]

    dev_f = set()
    while len(dev_f) < corpus_size: # using this because false gets too big
        dev_f.add(random.randrange(Fs[min_depth - 1], Fs[max_depth]))

    dev_f = list(dev_f)

    probs, suspects = zip(*(prob(x) for x in idx_t))

    project_root = Path(__file__).resolve().parent
    out_dir = project_root/"dataset"/args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path1 = out_dir / "train.pkl"
    out_path2 = out_dir / "dev_t.pkl"
    out_path3 = out_dir / "dev_f.pkl" 
    out_path4 = out_dir / "probs.pkl" 
    out_path5 = out_dir / "act_world.pkl"
    out_path6 = out_dir / "alt_worlds.pkl"
    out_path7 = out_dir / "params.pkl"


    with open(out_path1, "wb") as f:
      pickle.dump(idx_t,f)

    with open(out_path2, "wb") as f:
      pickle.dump(dev_t,f)

    with open(out_path3, "wb") as f:
      pickle.dump(list(dev_f),f)

    with open(out_path4, "wb") as f:
      pickle.dump(probs,f)

    with open(out_path5, "wb") as f:
      pickle.dump(act_world,f)

    with open(out_path6, "wb") as f:
      pickle.dump(alt_worlds,f)

    with open(out_path7, "wb") as f:
      pickle.dump((number_pl, min_depth, max_depth, corpus_size, prop_tf, n_worlds), f)

    print(f"Saved {len(idx_t)} formulas to {out_path1}")
    print(f"Saved {len(dev_t)} formulas to {out_path2}")
    print(f"Saved {len(dev_f)} formulas to {out_path3}")
    print(f"Saved {len(probs)} probabilities to {out_path4}")
    print(f"Saved act_world to {out_path5}")
    print(f"Saved alt_worlds to {out_path6}")
    print(f"Saved parameters to {out_path7}")

    # TODO: RESOLVER CASO TAUTOLOGIAS
    
if __name__ == "__main__":
    main()

