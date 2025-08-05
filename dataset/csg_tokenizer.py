
import numpy as np

# Token type IDs
TOKEN_TYPE = {
    "intersect": 0,
    "union": 1,
    "difference": 2,
    "halfspace": 3,
    "start_cell": 4,
    "end_cell": 5
}

# Surface axis/type mapping
SURFACE_TYPE_TO_AXIS = {"x-plane": 0, "y-plane": 1, "z-plane": 2}
SURFACE_TYPE_ID = {"x-plane": 0, "y-plane": 1, "z-plane": 2}
PAD_VAL = -1

def quantize(val, min_val=-1.0, max_val=1.0, bins=256):
    clipped = np.clip(val, min_val, max_val)
    scaled = (clipped - min_val) / (max_val - min_val)
    return int(round(scaled * (bins - 1)))

def parse_region(region_str):
    region_str = region_str.replace("(", " ( ").replace(")", " ) ").split()
    stack = []

    def is_operator(tok):
        return tok == "|"

    def parse_token(tok):
        sign = 1
        if tok.startswith('-'):
            sign = 0
            tok = tok[1:]
        return {"type": "halfspace", "id": int(tok), "sign": sign}

    def reduce_stack(s):
        if '|' in s:
            parts = [x for x in s if x != '|']
            return {"type": "union", "children": parts}
        else:
            return {"type": "intersect", "children": s}

    i = 0
    while i < len(region_str):
        tok = region_str[i]
        if tok == '(':
            stack.append(tok)
        elif tok == ')':
            tmp = []
            while stack and stack[-1] != '(':
                tmp.insert(0, stack.pop())
            stack.pop()
            stack.append(reduce_stack(tmp))
        elif is_operator(tok):
            stack.append(tok)
        else:
            stack.append(parse_token(tok))
        i += 1
    return stack[0] if stack else {}

def region_to_tokens(tree, surface_dict, parent_id=0):
    tokens = []
    if tree["type"] in ["intersect", "union"]:
        child_tokens = []
        for child in tree["children"]:
            child_tokens += region_to_tokens(child, surface_dict, parent_id=parent_id)
        tokens += child_tokens
        tokens.append([TOKEN_TYPE[tree["type"]]] + [PAD_VAL]*7)
    elif tree["type"] == "halfspace":
        sid = tree["id"]
        info = surface_dict.get(sid)
        if not info:
            return []
        surf_type = info["type"]
        coeff = info["coeffs"][0]
        surf_type_id = SURFACE_TYPE_ID.get(surf_type, -1)
        axis_id = SURFACE_TYPE_TO_AXIS.get(surf_type, -1)
        coeff_bin = quantize(coeff, -1.0, 1.0, 256)
        token = [
            TOKEN_TYPE["halfspace"],
            surf_type_id,
            tree["sign"],
            axis_id,
            coeff_bin,
            sid,
            parent_id,
            PAD_VAL
        ]
        tokens.append(token)
    return tokens

def tokens_to_region_v3(tokens):
    stack = []
    groups = []
    current_group = []

    for t in tokens:
        t_type = t[0]
        if t_type == TOKEN_TYPE["halfspace"]:
            sid = t[5]
            sign = t[2]
            lit = f"-{sid}" if sign == 0 else f"{sid}"
            current_group.append(lit)
        elif t_type == TOKEN_TYPE["intersect"]:
            if current_group:
                groups.append(f"({' '.join(current_group)})")
                current_group = []
        elif t_type == TOKEN_TYPE["union"]:
            if current_group:
                groups.append(f"({' '.join(current_group)})")
                current_group = []
    if current_group:
        groups.append(f"({' '.join(current_group)})")
    return f"({' | '.join(groups)})" if len(groups) > 1 else groups[0]
