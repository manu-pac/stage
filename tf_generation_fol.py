from classes_fol import V, P, InterpretationFunc, Model, Formula, PredApp, Neg, Ex, Conj
import argparse
import string
import random
import itertools
import pickle
from pathlib import Path

def pred_generator(predicates, min_arity, max_arity):
    #determines the arity of the predicates and returns a list of predicate objects
    pred_list = []
    for pred in predicates:
        arity = random.randint(min_arity, max_arity)
        pred_list.append(P(pred, arity))
    pred_list.sort(key=lambda obj: obj.arity)
    return pred_list

def form(i, d, bag):
    # bag: list of V·s that are currently "available" (already bound)
    k = len(bag) 
    cache_key = (i, d, tuple(bag))
    if cache_key in cache:
        return cache[cache_key]

    if d == 1: # blocks of predicates vs. variable combos (one block per arity group)
        cur_arity = 1
        cur_var_combos = list(itertools.product(bag, repeat=cur_arity)) if k > 0 else [] # variables drawn only from the bag
        # poss_combos = how many predicates with this arity * how many bound variables combos for this arity 
        poss_combos = len(cur_var_combos) * sum([(it.arity == cur_arity) for it in p])
        pred_idx = 0

        while i + 1 > poss_combos: # if idx is bigger than arity 1, repeat for next arity until it's right
            i = i - poss_combos
            pred_idx += sum([(it.arity == cur_arity) for it in p])
            cur_arity += 1
            cur_var_combos = list(itertools.product(bag, repeat=cur_arity)) if k > 0 else []
            poss_combos = len(cur_var_combos) * sum([(it.arity == cur_arity) for it in p])

        pred, var = divmod(i, len(cur_var_combos))
        phi = PredApp(p[pred + pred_idx], list(cur_var_combos[var]))

    elif i < C[d - 1][k]: #inside negation block (same bag, depth d-1)
        phi = Neg(form(i, d - 1, bag))

    else:
        j = i - C[d - 1][k] #update index to align it w/ ex. group

        size_ex_old = k * C[d - 1][k] # "redundant" ex. case (reintroduces bounr var.)
        size_ex_new = (n_vars - k) * C[d - 1].get(k + 1, 0) # bind one of the n-k fresh vars (bag grows to k+1)
        sizeEx = size_ex_old + size_ex_new

        if j < sizeEx: # inside ex. block
            if j < size_ex_old:
                var_idx, frm = divmod(j, C[d - 1][k])
                phi = Ex(bag[var_idx], form(frm, d - 1, bag)) # reintroducing already bound var, bag unchanged
            else:
                j2 = j - size_ex_old
                new_vars = [x for x in v if x not in bag]
                var_idx, frm = divmod(j2, C[d - 1][k + 1]) 
                phi = Ex(new_vars[var_idx], form(frm, d - 1, bag + [new_vars[var_idx]])) # new bound var, added to bag

        else: # first conjunction group = deep^anything (both sides see the same bag)
            j -= sizeEx
            sizeA = C[d - 1][k] * S[d - 1][k]

            if j < sizeA:
                row, col = divmod(j, S[d - 1][k])
                phi = Conj(form(row, d - 1, bag), form_le(col, k, bag))

            else: #second conjunction group = shallow^deep
                j -= sizeA
                row, col = divmod(j, C[d - 1][k])
                phi = Conj(form_le(row, k, bag), form(col, d - 1, bag))

    cache[cache_key] = phi
    return phi

def form_le(i, k, bag):
    e = 1
    while i >= C[e][k]:
        i -= C[e][k]
        e += 1
    return form(i, e, bag)

def true_le(i):
    # just to match with training code
    return str(form_le(i, 0, []))

def false_le(i):
    #same
    return str(form_le(i, 0, []))

def _index_to_tuple(idx, domain, arity):
    # decode a flat index in [0, len(domain)**arity) into a domain-tuple (mixed-radix)
    base = len(domain)
    digits = []
    for _ in range(arity):
        idx, r = divmod(idx, base)
        digits.append(domain[r])
    return tuple(reversed(digits))

def generate_worlds(predicates, domain, n_worlds, prop_tf):
    all_worlds = set()
    max_attempts = n_worlds * 10
    attempts = 0

    while len(all_worlds) < n_worlds and attempts < max_attempts:
        attempts += 1
        world_items = []
        for pred in predicates:
            arity = pred.arity
            total_tuples = len(domain) ** arity
            n_true = random.randint(2,int(total_tuples * prop_tf)) if prop_spread else int(total_tuples * prop_tf)
            sampled_indices = random.sample(range(total_tuples), n_true)
            true_tuples = frozenset(_index_to_tuple(i, domain, arity) for i in sampled_indices)

            world_items.append((pred, true_tuples))

        all_worlds.add(frozenset(world_items))

    return list(all_worlds)

def setup(domain_, variables_, predicates_, min_arity_, max_arity_, min_depth_, max_depth_, act_world_, alt_worlds_):
    global domain, variables, p, v, n_vars, min_depth, max_depth, act_world, alt_worlds, C, S, cache

    domain = domain_
    variables = variables_
    p = predicates_
    v = [V(name) for name in variables_]
    n_vars = len(v)
    min_depth = min_depth_
    max_depth = max_depth_
    act_world = act_world_
    alt_worlds = alt_worlds_

    # rebuild the index tables C and S, just like in main()
    ks = range(0, n_vars + 1)
    C = {1: {k: sum(k ** pr.arity for pr in p) for k in ks}}
    S = {0: {k: 0 for k in ks}, 1: dict(C[1])}

    for d in range(2, max_depth + 1):
        C[d] = {}
        for k in ks:
            c_prev_next_k = C[d - 1][k + 1] if (k + 1) in C[d - 1] else 0
            neg = C[d - 1][k]
            ex = k * C[d - 1][k] + (n_vars - k) * c_prev_next_k
            conj = C[d - 1][k] * S[d - 1][k] + S[d - 2][k] * C[d - 1][k]
            C[d][k] = neg + ex + conj
        S[d] = {k: S[d - 1][k] + C[d][k] for k in ks}

    cache = {} # clear the formula cache because we have a new set of worlds

def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_id", type=int, required=True, help="number of individuals")
    parser.add_argument("--number_vr", type=int, required=True, help="number of variables")
    parser.add_argument("--number_pr", type=int, required=True, help="number of predicates")
    parser.add_argument("--min_depth", type=int, required=True, help="minimum depth of formulas")
    parser.add_argument("--max_depth", type=int, required=True, help="maximum depth of formulas")
    parser.add_argument("--min_arity", type=int, default=1, help="minimum arity of the predicates")
    parser.add_argument("--max_arity", type=int, required=True, help="maximum arity of the predicates")
    parser.add_argument("--corpus_size", type=int, required=True)
    parser.add_argument("--n_worlds", type=int, required=True)
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--prop_tf", type=float, default=0.5)
    parser.add_argument('--prop_spread', action='store_true', default=False, help="predicates will pick BETWEEN 10 and prop_tf percent of the available tuples (varies by predicate)")
    args = parser.parse_args()

    global number_id, number_vr, number_pr, min_depth, max_depth, corpus_size, n_worlds, min_arity, max_arity, domain, variables, predicates, p
    global cache, v, n_vars, act_world, alt_worlds, prop_spread

    number_id = args.number_id
    number_vr = args.number_vr
    number_pr = args.number_pr
    min_depth = args.min_depth
    max_depth = args.max_depth
    corpus_size = args.corpus_size
    n_worlds = args.n_worlds
    min_arity = args.min_arity
    max_arity = args.max_arity
    prop_spread = args.prop_spread

    # define domain, variables, predicates (shared between worlds)
    domain = list(range(number_id))
    variables = list(string.ascii_lowercase[:number_vr])
    predicates = list(string.ascii_uppercase[:number_pr])
    p = pred_generator(predicates, min_arity, max_arity)
    v = [V(name) for name in variables]
    n_vars = len(v)

    # generate all worlds (actual + alternatives)
    worlds_list = generate_worlds(p, domain, n_worlds, prop_tf=args.prop_tf)
    act_world = worlds_list[0]
    alt_worlds = worlds_list[1:]

    # initialise tables and cache via setup()
    setup(domain, variables, p, min_arity, max_arity,
          min_depth, max_depth, act_world, alt_worlds)

    #build actual world model for truth checking
    p_dict = {pred: set(tuples) for pred, tuples in act_world}
    i_func = InterpretationFunc(set(), p_dict)
    model = Model(set(domain), i_func)

    # collect training & dev indices (true/false), with min-depth filtering
    start_idx = S[min_depth - 1][0] if min_depth > 1 else 0
    total_idx = S[max_depth][0]

    if corpus_size > (total_idx - start_idx):
        raise ValueError(f"corpus_size ({corpus_size}) exceeds available closed formulas with depth in [{min_depth}, {max_depth}] ({total_idx - start_idx}).")

    seen = set()
    train_true = []
    dev_true = []
    dev_false = []
    attempts = 0
    max_attempts = 10_000_000

    print(f"Generating and filtering formulas (depth {min_depth}–{max_depth}, total indices {total_idx})...")

    while (len(train_true) < corpus_size or len(dev_true) < corpus_size or len(dev_false) < corpus_size):
        if attempts > max_attempts:
            raise RuntimeError("Too many attempts. Try increasing max_depth or decreasing corpus_size.")
        idx = random.randint(start_idx, total_idx - 1)
        if idx in seen:
            attempts += 1
            continue
        seen.add(idx)

        if form_le(idx, 0, []).check_closed(model):
            if len(train_true) < corpus_size:
                train_true.append(idx)
            elif len(dev_true) < corpus_size:
                dev_true.append(idx)
        else:
            if len(dev_false) < corpus_size:
                dev_false.append(idx)

        attempts += 1

    print(f"Collected: train_true={len(train_true)}, dev_true={len(dev_true)}, dev_false={len(dev_false)}")

    #compute probs over alternative worlds (same as PL)
    print("Computing probabilities over alternative worlds...")

    alt_models = []
    for alt_w in alt_worlds:
        alt_dict = {pred: set(tuples) for pred, tuples in alt_w}
        alt_i = InterpretationFunc(set(), alt_dict)
        alt_models.append(Model(set(domain), alt_i))

    formula_cache = {}
    for idx in train_true:
        formula_cache[idx] = form_le(idx, 0, [])

    # Now evaluate
    probs = []
    for idx in train_true:
        phi = formula_cache[idx]
        false_count = 0
        for model in alt_models:
            if not phi.check_closed(model):
                false_count += 1
        probs.append(false_count / len(alt_worlds))

    #save everything to disk
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "dataset" / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.pkl", "wb") as f:
        pickle.dump(train_true, f)
    with open(out_dir / "dev_t.pkl", "wb") as f:
        pickle.dump(dev_true, f)
    with open(out_dir / "dev_f.pkl", "wb") as f:
        pickle.dump(dev_false, f)
    with open(out_dir / "probs.pkl", "wb") as f:
        pickle.dump(probs, f)
    with open(out_dir / "act_world.pkl", "wb") as f:
        pickle.dump(act_world, f)
    with open(out_dir / "alt_worlds.pkl", "wb") as f:
        pickle.dump(alt_worlds, f)

    # save parameters as a dict with all needed fields
    params = {
        'logic': 'fol',
        'domain': domain,
        'variables': variables,
        'predicates': [(pr._name, pr.arity) for pr in p],
        'number_pr': number_pr,
        'number_vr': number_vr,
        'min_arity': min_arity,
        'max_arity': max_arity,
        'min_depth': min_depth,
        'max_depth': max_depth,
        'corpus_size': corpus_size,
        'n_worlds': n_worlds,
        'args': vars(args) # keep all original CLI args for reference
    }
    with open(out_dir / "params.pkl", "wb") as f:
        pickle.dump(params, f)

    print(f"Saved {len(train_true)} training indices to {out_dir / 'train.pkl'}")
    print(f"Saved {len(dev_true)} dev_true indices to {out_dir / 'dev_t.pkl'}")
    print(f"Saved {len(dev_false)} dev_false indices to {out_dir / 'dev_f.pkl'}")
    print(f"Saved {len(probs)} probabilities to {out_dir / 'probs.pkl'}")
    print(f"Saved act_world to {out_dir / 'act_world.pkl'}")
    print(f"Saved {len(alt_worlds)} alt_worlds to {out_dir / 'alt_worlds.pkl'}")
    print(f"Saved parameters to {out_dir / 'params.pkl'}")
    print("Done.")

if __name__ == "__main__":
    main()