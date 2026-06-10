# Source Generated with Decompyle++
# File: validate_ghost_teacher.cpython-311.pyc (Python 3.11)

"""Ghost Teacher validation: does the transcoder reconstruct inside TransformerLens?

If reconstruction here is good (cosine > 0.9) while it failed on raw HF activations,
that proves the transcoders need TL-native activations, and the Ghost Teacher
architecture (generate y_target in TL, attribute gradients in HF) is justified.

We load TL with the *NousResearch* weights via hf_model= (meta-llama is gated),
matching the fork's processing: fold_ln=False, center_writing_weights=False.
"""
from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from transcoder_io import load_config, load_layer_transcoder

def main():
    pass
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
