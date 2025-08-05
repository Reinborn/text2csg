
import os
import numpy as np
import torch
from csg_tokenizer_with_offset import parse_region, region_to_tokens_with_depth, SURFACE_TYPE_ID, SURFACE_TYPE_TO_AXIS, SURFACE_ID_OFFSET, PAD_VAL
import xml.etree.ElementTree as ET

def extract_surface_table(xml_path):
    surface_dict = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for s in root.findall("surface"):
        sid = int(s.attrib["id"])
        coeffs = list(map(float, s.attrib["coeffs"].split()))
        surface_dict[sid] = {"type": s.attrib["type"], "coeffs": coeffs}
    return surface_dict

def convert_xml_to_tokens(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    surface_dict = extract_surface_table(xml_path)
    all_token_seqs = []

    for cell in root.findall("cell"):
        region_str = cell.attrib["region"]
        region_tree = parse_region(region_str)
        tokens = []
        tokens.append([4] + [PAD_VAL]*7)  # start_cell
        tokens += region_to_tokens_with_depth(region_tree, surface_dict, parent_id=0, current_depth=0, next_id=[1])
        tokens.append([5] + [PAD_VAL]*7)  # end_cell
        all_token_seqs.append(tokens)

    return all_token_seqs

def run_pipeline(xml_path, output_token_path="tokens.npy"):
    token_seqs = convert_xml_to_tokens(xml_path)
    np.save(output_token_path, np.array(token_seqs, dtype=object))
    print(f"✅ Saved tokens to {output_token_path}")
    print("🚀 Starting training...")
    os.system("python train_csg.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="Path to input MCNP-style CSG XML file")
    parser.add_argument("--output", type=str, default="tokens.npy", help="Output path for token array")
    args = parser.parse_args()

    run_pipeline(args.xml, args.output)
