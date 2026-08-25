import pickle
from pathlib import Path
import numpy as np
from classes_fol import PredApp, Neg, Ex, Conj, V, P, InterpretationFunc, Model, VarAssignment
from tf_generation_fol import form_le, setup
import argparse
from collections import defaultdict

def convert_to_plotting_format(filtered_dev):
    grouped = defaultdict(list)
    for rec in filtered_dev:
        grouped[rec["idx"]].append((rec["char_idx"], rec["witness"]))
    return list(grouped.items())  # -> [(idx, [(char_idx, witness), ...]), ...]

_fresh_counter = [0]
def fresh_var():
    _fresh_counter[0] += 1
    return V(f"__occ_{_fresh_counter[0]}__")

def enumerate_occurrences_tagged(formula):
    if isinstance(formula, PredApp):
        s = f"{formula._pred}({','.join([str(x) for x in formula._args])})"
        prefix = f"{formula._pred}("
        variants, occ_positions = [], []
        pos = len(prefix)
        for i, arg in enumerate(formula._args):
            if isinstance(arg, V):
                fv = fresh_var()
                new_args = list(formula._args)
                new_args[i] = fv
                variants.append((PredApp(formula._pred, new_args), fv, arg))
                occ_positions.append((fv, pos))
            pos += len(str(arg)) + 1
        return variants, s, occ_positions

    elif isinstance(formula, Neg):
        sub_variants, sub_str, sub_occ = enumerate_occurrences_tagged(formula._phi)
        prefix = '(¬'
        offset = len(prefix)
        s = prefix + sub_str + ')'
        variants = [(Neg(v), fv, ov) for v, fv, ov in sub_variants]
        occ_positions = [(fv, idx + offset) for fv, idx in sub_occ]
        return variants, s, occ_positions

    elif isinstance(formula, Ex):
        sub_variants, sub_str, sub_occ = enumerate_occurrences_tagged(formula._phi)
        prefix = f'(∃{formula._v}'
        offset = len(prefix)
        s = prefix + sub_str + ')'
        variants = [(Ex(formula._v, v), fv, ov) for v, fv, ov in sub_variants]
        occ_positions = [(fv, idx + offset) for fv, idx in sub_occ]
        return variants, s, occ_positions

    elif isinstance(formula, Conj):
        left_variants, left_str, left_occ = enumerate_occurrences_tagged(formula._phi)
        right_variants, right_str, right_occ = enumerate_occurrences_tagged(formula._psi)
        left_offset = 1
        right_offset = left_offset + len(left_str) + 1
        s = f'({left_str}∧{right_str})'
        variants = [(Conj(v, formula._psi), fv, ov) for v, fv, ov in left_variants]
        variants += [(Conj(formula._phi, v), fv, ov) for v, fv, ov in right_variants]
        occ_positions = [(fv, idx + left_offset) for fv, idx in left_occ]
        occ_positions += [(fv, idx + right_offset) for fv, idx in right_occ]
        return variants, s, occ_positions

    else:
        raise TypeError(f"Unhandled formula type: {type(formula)}")


def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True, help="name of the folder of the dataset")
    parser.add_argument("--target_count", type=int, default=1500,
                         help="number of singleton-profile VARIABLE OCCURRENCES to collect (not formulas)")
    args = parser.parse_args()

    folder_name = args.folder_name
    target_count = args.target_count

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "dataset" / folder_name

    with open(data_dir / "params.pkl", "rb") as f:
        params = pickle.load(f)

    domain = params['domain']
    variables = params['variables']
    predicates_info = params['predicates']

    predicates = [P(name, arity) for name, arity in predicates_info]

    with open(data_dir / "act_world.pkl", "rb") as f:
        act_world = pickle.load(f)

    p_dict = {pred: set(tuples) for pred, tuples in act_world}
    i_func = InterpretationFunc({}, p_dict)
    model = Model(set(domain), i_func)

    with open(data_dir / "dev_t.pkl", "rb") as f:
        dev_true_indices = pickle.load(f)

    setup(domain, variables, predicates,
          params['min_arity'], params['max_arity'],
          params['min_depth'], params['max_depth'],
          act_world, [])

    print("variables from params:", variables)

    import time

    print("len(dev_true_indices):", len(dev_true_indices))
    print("len(set(dev_true_indices)):", len(set(dev_true_indices)))

    # one entry per singleton-profile OCCURRENCE (this is what target_count counts)
    filtered_dev = []

    total_processed = 0
    n_passes = 0
    t0 = time.time()

    while len(filtered_dev) < target_count:
        n_passes += 1
        added_this_pass = 0
        pass_seen = set()
        stop = False

        for idx in dev_true_indices:
            if total_processed % 20 == 0:
                print(f"  ...processing formula #{total_processed} (idx={idx}), "
                      f"singletons found so far: {len(filtered_dev)}, "
                      f"elapsed = {time.time()-t0:.1f}s", flush=True)
            pass_seen.add(idx)
            total_processed += 1

            f = form_le(idx, 0, [])
            variants, full_str, occ_positions = enumerate_occurrences_tagged(f)
            pos_lookup = dict(occ_positions)

            for variant, fv, orig_var in variants:
                profile = set()
                for i in domain:
                    assignment = VarAssignment({fv: i})
                    if variant.check(model, assignment):
                        profile.add(i)

                if len(profile) == 1:
                    filtered_dev.append({
                        "idx": idx,
                        "formula_str": full_str,
                        "char_idx": pos_lookup[fv],
                        "orig_var": str(orig_var),
                        "witness": next(iter(profile)),
                    })
                    added_this_pass += 1
                    if len(filtered_dev) >= target_count:
                        stop = True
                        break

            if stop:
                break

        print(f"pass {n_passes}: distinct idx seen this pass = {len(pass_seen)}, "
            f"singleton occurrences added this pass = {added_this_pass}, "
            f"total formulas processed so far = {total_processed}, "
            f"filtered_dev so far = {len(filtered_dev)}, elapsed = {time.time()-t0:.1f}s")

        if added_this_pass == 0:
            print("No new singleton occurrences found this pass: stopping to avoid infinite loop.")
            break

    #convert to format used in next plot
    filtered_dev = convert_to_plotting_format(filtered_dev)

    print(f"TOTAL: {total_processed} formulas processed across {n_passes} pass(es), "
        f"{len(filtered_dev)} singleton occurrences kept, {time.time()-t0:.1f}s elapsed")

    with open(data_dir / "dev_data.pkl", "wb") as f:
        pickle.dump(filtered_dev, f)

    print(f"Saved {len(filtered_dev)} singleton occurrence records to {data_dir / 'dev_data.pkl'}")


if __name__ == "__main__":
    main()