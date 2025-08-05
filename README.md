# CSG2CAD

CSG2CAD is a structure-aware neural generation framework that translates natural language descriptions into valid Constructive Solid Geometry (CSG) programs.

This project extends and reimplements components of DeepCAD and Text2CAD, but features a completely custom tokenization, vectorization, and decoding pipeline designed for symbolic and hierarchical CSG modeling.

## ✨ Key Features

- CSG Tokenizer and Token Embedding
- Tree-Guided Attention with Parent and Sibling Masks
- Structure-aware Loss Functions
- Text-to-CSG Integration Pipeline


## 📦 Dataset & Supplementary Files

This repository includes two ZIP archives that support training and evaluation:

| File | Description |
|------|-------------|
| `csg_dataset.zip` | A symbolic CSG dataset containing structured geometry programs in XML format. These are parsed into tokens and used as ground-truth for training the CSG decoder. |
| `text2csg_content.zip` | A collection of natural language descriptions paired with CSG models, useful for training the text-conditioned generation pipeline. |


