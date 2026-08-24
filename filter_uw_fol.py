import pickle
from pathlib import Path
import numpy as np
from classes_fol import PredApp, Neg, Ex, Conj, V, P, InterpretationFunc, Model, VarAssignment
from tf_generation_fol import form_le, setup
import argparse

def remove_one_ex_tagged(formula):
    if isinstance(formula, PredApp):
        return [], str(formula), []

    elif isinstance(formula, Neg):
        sub_variants, sub_str, sub_binders = remove_one_ex_tagged(formula._phi)
        prefix = '(¬'
        offset = len(prefix)
        s = prefix + sub_str + ')'
        variants = [(Neg(v), removed) for v, removed in sub_variants]
        binders = [(node, idx + offset) for node, idx in sub_binders]
        return variants, s, binders

    elif isinstance(formula, Ex):
        sub_variants, sub_str, sub_binders = remove_one_ex_tagged(formula._phi)
        prefix = f'(∃{formula._v}'
        offset = len(prefix)
        bound_idx = offset - len(str(formula._v))
        s = prefix + sub_str + ')'

        variants = [(formula._phi, formula)]  # option 1: remove this quantifier
        variants += [(Ex(formula._v, sub), removed) for sub, removed in sub_variants]  # option 2: recurse

        binders = [(formula, bound_idx)] + [(node, idx + offset) for node, idx in sub_binders]
        return variants, s, binders

    elif isinstance(formula, Conj):
        left_variants, left_str, left_binders = remove_one_ex_tagged(formula._phi)
        right_variants, right_str, right_binders = remove_one_ex_tagged(formula._psi)

        left_offset = 1  # '('
        right_offset = left_offset + len(left_str) + 1  # + '∧'
        s = f'({left_str}∧{right_str})'

        variants = [(Conj(lv, formula._psi), removed) for lv, removed in left_variants]
        variants += [(Conj(formula._phi, rv), removed) for rv, removed in right_variants]

        binders = [(node, idx + left_offset) for node, idx in left_binders]
        binders += [(node, idx + right_offset) for node, idx in right_binders]
        return variants, s, binders

    else:
        raise TypeError(f"Unhandled formula type: {type(formula)}")

def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True, help="name of the folder of the dataset")
    parser.add_argument("--target_count", type=int, default=1500, help="number of formulas to keep after filtering")
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

    v_objs = [V(i) for i in params['variables']]

    # Reconstruct predicate objects
    predicates = [P(name, arity) for name, arity in predicates_info]

    # Load the actual world
    with open(data_dir / "act_world.pkl", "rb") as f:
        act_world = pickle.load(f)  # list of (pred, frozenset of tuples)

    # Build the interpretation function
    p_dict = {pred: set(tuples) for pred, tuples in act_world}
    i_func = InterpretationFunc({}, p_dict)  # no constants
    model = Model(set(domain), i_func)

    with open(data_dir / "dev_t.pkl", "rb") as f:
        dev_true_indices = pickle.load(f)

    setup(domain, variables, predicates,
          params['min_arity'], params['max_arity'],
          params['min_depth'], params['max_depth'],
          act_world, [])

    print("variables from params:", variables)
    
    global s_dict

    #dict of dicts: every variable assignment
    s_dict = {}

    for i in domain:
        s_dict[i] = {}
        for var in v_objs:
            s_dict[i][var]=i

    filtered_dev = []

    while len(filtered_dev) < target_count:
        for idx in dev_true_indices:
            f = form_le(idx, 0, [])
            variants, full_str, binder_indices = remove_one_ex_tagged(f)
            binder_lookup = dict(binder_indices)

            unique_witness_results = []  # list of (var_name, char_idx, witness)

            for variant, removed_node in variants:
                true_count = 0
                witness = None
                for i in domain:
                    if variant.check(model, VarAssignment(s_dict[i])):
                        true_count += 1
                        witness = i
                        if true_count > 1:
                            break

                if true_count == 1:
                    char_idx = binder_lookup[removed_node]
                    unique_witness_results.append((char_idx, witness)) #MUDAR AQUI SE QUISER LETRA

            if unique_witness_results:
                filtered_dev.append((idx, unique_witness_results))
                print(len(filtered_dev))
                if len(filtered_dev) >= target_count:
                    break
    
    # save filtered data
    with open(data_dir / "dev_data.pkl", "wb") as f:
        pickle.dump(filtered_dev, f)

    print(f"Filtered dev_t: {len(filtered_dev)} formulas kept (out of {len(dev_true_indices)})")
    print(f"Saved to {data_dir / 'dev_data.pkl'}")


if __name__ == "__main__":
    main()