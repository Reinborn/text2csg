
from lxml import etree

TOKEN_TYPE = {
    0: "intersect",
    1: "union",
    2: "difference",
    3: "halfspace",
    4: "start_cell",
    5: "end_cell"
}

SURFACE_ID_OFFSET = 10

def tokens_to_region(tokens):
    stack = []
    for t in tokens:
        if t[0] == 3:  # halfspace
            sid = t[5] - SURFACE_ID_OFFSET
            if t[2] == 0:
                sid = -sid
            stack.append(str(sid))
        elif t[0] == 0:  # intersect
            children = []
            while stack and isinstance(stack[-1], str):
                children.insert(0, stack.pop())
            stack.append(f"( {' '.join(children)} )")
        elif t[0] == 1:  # union
            children = []
            while stack and isinstance(stack[-1], str):
                children.insert(0, stack.pop())
            joined = ' | '.join(children)
            stack.append(f"( {joined} )")
    return stack[-1] if stack else ""

def tokens_to_xml(tokens, surfaces, cell_ids=[0, 1], universe="1"):
    root = etree.Element("geometry")

    # Build region expressions
    region = tokens_to_region(tokens)
    for cid in cell_ids:
        cell = etree.Element("cell", id=str(cid), material="void", region=region, universe=universe)
        root.append(cell)

    # Add surfaces
    for sid, s in surfaces.items():
        s_elem = etree.Element("surface", id=str(sid), type=s["type"])
        s_elem.set("coeffs", s["coeffs"])
        if "boundary" in s:
            s_elem.set("boundary", s["boundary"])
        root.append(s_elem)

    return etree.tostring(root, pretty_print=True, encoding="UTF-8", xml_declaration=True).decode("utf-8")
