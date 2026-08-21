import pickle
from pathlib import Path
import numpy as np
from classes_fol import PredApp, Neg, Ex, Conj, V, P, InterpretationFunc, Model
from tf_generation_fol import form_le, setup
import argparse

def get_variables(phi):
    if isinstance(phi, PredApp):
        return set(phi._args)
    if isinstance(phi, Neg):
        return get_variables(phi._phi)
    if isinstance(phi, Conj):
        return get_variables(phi._phi) | get_variables(phi._psi)
    if isinstance(phi, Ex):
        return {phi._v} | get_variables(phi._phi)
    raise TypeError(f"Unknown type: {type(phi)}")


def get_matrix(phi):
    if isinstance(phi, PredApp):
        return phi
    if isinstance(phi, Neg):
        return Neg(get_matrix(phi._phi))
    if isinstance(phi, Conj):
        return Conj(get_matrix(phi._phi), get_matrix(phi._psi))
    if isinstance(phi, Ex):
        return get_matrix(phi._phi)
    raise TypeError(f"Unknown type: {type(phi)}")


def _build_relation_cache(model):
    domain_size = len(model.domain)

    cache = {}
    for pred, tuples in model.i_func._p_dic.items():
        rel = np.zeros((domain_size,) * pred.arity, dtype=bool)
        for tup in tuples:
            rel[tup] = True
        cache[pred] = rel

    return cache, domain_size

def _binder_positions(phi):
    positions = {}
    parts = []
    pos = 0

    def emit(s):
        nonlocal pos
        parts.append(s)
        pos += len(s)

    def walk(node):
        if isinstance(node, PredApp):
            emit(str(node))
        elif isinstance(node, Neg):
            emit("(¬")
            walk(node._phi)
            emit(")")
        elif isinstance(node, Ex):
            emit("(∃")
            positions[node._v] = pos
            emit(str(node._v))
            walk(node._phi)
            emit(")")
        elif isinstance(node, Conj):
            emit("(")
            walk(node._phi)
            emit("∧")
            walk(node._psi)
            emit(")")
        else:
            raise TypeError(f"Unknown type: {type(node)}")

    walk(phi)
    assert "".join(parts) == str(phi), "_binder_positions drifted from __str__ -- check class formatting"
    return positions


def compute_unique_mapping_fast(phi, model, rel_cache, domain_size, max_vars=None, grid_cache=None):
    matrix = get_matrix(phi)
    vars_list = list(get_variables(matrix))
    n = len(vars_list)

    if not vars_list:
        return {} if matrix.check_closed(model) else None
    if max_vars is not None and n > max_vars:
        return None

    var_to_axis = {v: i for i, v in enumerate(vars_list)}
    shape = (domain_size,) * n

    if grid_cache is not None and n in grid_cache:
        grids = grid_cache[n]
    else:
        grids = np.ix_(*([range(domain_size)] * n))
        if grid_cache is not None:
            grid_cache[n] = grids

    def get_relation(pred):
        return rel_cache.setdefault(pred, np.zeros((domain_size,) * pred.arity, dtype=bool))

    def eval_node(node):
        if isinstance(node, PredApp):
            idx = tuple(grids[var_to_axis[arg]] for arg in node._args)
            return get_relation(node._pred)[idx]
        if isinstance(node, Neg):
            return ~eval_node(node._phi)
        if isinstance(node, Conj):
            return eval_node(node._phi) & eval_node(node._psi)
        raise TypeError(f"Unexpected node: {type(node)}")

    mask = np.broadcast_to(eval_node(matrix), shape)
    if not mask.any():
        return None

    unique_map = {}
    for var, axis in var_to_axis.items():
        other_axes = tuple(a for a in range(n) if a != axis)
        appears = np.any(mask, axis=other_axes)
        idxs = np.flatnonzero(appears)
        if idxs.size == 1:
            unique_map[var] = int(idxs[0])

    return unique_map


def get_unique_variable_positions(phi, model, rel_cache, domain_size, max_vars=None, grid_cache=None):
    um = compute_unique_mapping_fast(phi, model, rel_cache, domain_size, max_vars=max_vars, grid_cache=grid_cache)
    if not um:
        return None

    positions = _binder_positions(phi)
    return {"unique_vars": {str(v): {"value": val, "position": positions[v]} for v, val in um.items()}}


def filter_formulas_with_positions(dev_true_indices, model, form_le, target_count=1500,
                                   max_vars=None, report_every=500):
    pool = list(dev_true_indices)

    rel_cache, domain_size = _build_relation_cache(model)
    grid_cache = {}

    filtered = []
    tried = 0
    for idx in pool:
        tried += 1
        phi = form_le(idx, 0, [])
        result = get_unique_variable_positions(phi, model, rel_cache, domain_size,
                                               max_vars=max_vars, grid_cache=grid_cache)
        if result is not None:
            filtered.append((idx, result))
            if len(filtered) >= target_count:
                break

        if report_every and tried % report_every == 0:
            print(f"tried {tried}, kept {len(filtered)}")

    if len(filtered) < target_count:
        print(f"WARNING: pool exhausted ({len(pool)} formulas) before reaching target_count={target_count}; "
              f"kept {len(filtered)}.")
    else:
        print(f"Kept {len(filtered)} formulas after trying {tried}/{len(pool)}")

    return filtered


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

    filtered_dev = filter_formulas_with_positions(dev_true_indices, model, form_le, target_count=target_count)

    # save filtered data
    with open(data_dir / "dev_data.pkl", "wb") as f:
        pickle.dump(filtered_dev, f)

    print(f"Filtered dev_t: {len(filtered_dev)} formulas kept (out of {len(dev_true_indices)})")
    print(f"Saved to {data_dir / 'dev_data.pkl'}")


if __name__ == "__main__":
    main()