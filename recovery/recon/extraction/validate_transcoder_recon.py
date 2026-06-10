# Source Generated with Decompyle++
# File: validate_transcoder_recon.cpython-311.pyc (Python 3.11)

"""Sanity check: does decode(encode(mlp_in)) reconstruct the MLP output?

A low relative error (<< 1.0) confirms two things at once:
  (1) the hook convention is right (mlp.hook_in == HF post_attention_layernorm out),
  (2) the (mirror) base model weights match the transcoder's training model.

Predicting zero gives rel error 1.0 by definition, so anything clearly below ~0.6
means the transcoder is genuinely reconstructing.
"""
from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transcoder_io import load_config, load_layer_transcoder

def main():
    pass
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
