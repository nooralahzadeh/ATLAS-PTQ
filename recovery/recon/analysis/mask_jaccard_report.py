# Source Generated with Decompyle++
# File: mask_jaccard_report.cpython-310.pyc (Python 3.10)

'''Jaccard diagnostics and v3 mask builders (anchored union + module split).

Compares TaCQ vs T-DSO outlier masks at matched 0.35% budget, emits layer×module
heatmaps, and constructs bitwidth-adaptive v3 masks:

  v3-e anchored union:
    M_core = M_TaCQ ∩ M_TDSO  (immutable shared core)
    fill remaining budget from XOR region ranked by dispute_bias scores

  v3-c module split:
    attn budget ranked by T-DSO saliency; MLP budget ranked by TaCQ saliency

Score tensors (recommended for dispute tie-breaking):
  --tacq-scores   TaCQ importances or |W|·g_ce·|ΔW| saliency .pt
  --tdso-scores   T-DSO combined saliency .pt

Mask-only fallback (coarser): XOR region ranked by module heuristic
  tdso bias → prefer tdso-unique + attention projections
  tacq bias → prefer tacq-unique + MLP projections
'''
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
ATTN_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP_SUFFIXES = ('gate_proj', 'up_proj', 'down_proj')
ALL_SUFFIXES = ATTN_SUFFIXES + MLP_SUFFIXES

def load_flat_mask(path = None):
    d = torch.load(path, 'cpu', False, **('map_location', 'weights_only'))
    if isinstance(d, dict) and 'masks' in d:
        d = d['masks']
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(d.items())


def load_scores(path = None):
    '''Load float saliency / importances keyed by weight name.'''
    d = torch.load(path, 'cpu', False, **('map_location', 'weights_only'))
    if isinstance(d, dict):
        for key in ('saliency', 'scores', 'g_ce_saliency', 'g_align_saliency'):
            if key in d and isinstance(d[key], dict):
                d = d[key]
            
            if 'masks' in d:
                raise ValueError(f'''{path} looks like a bool mask; pass --tacq-mask / --tdso-mask instead''')
            out = { }
            for k, v in d.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.dtype == torch.bool:
                    out[k] = v.float()
                    continue
                out[k] = v.float().abs()
            return out


def layer_idx(name = None):
    m = re.search('layers\\.(\\d+)\\.', name)
    if m:
        return int(m.group(1))


def component(name = None):
    for suf in ALL_SUFFIXES:
        if name.endswith(suf + '.weight'):
            return suf
        return 'other'


def is_attn(name = None):
    return None((lambda .0 = None: for s in .0:
name.endswith(s + '.weight'))(ATTN_SUFFIXES))


def is_mlp(name = None):
    return None((lambda .0 = None: for s in .0:
name.endswith(s + '.weight'))(MLP_SUFFIXES))


def align_keys(a = None, b = None):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit('No shared weight keys between inputs')
    return keys


def count_kept(mask = None):
    return int(sum((lambda .0: for v in .0:
int(v.sum()))(mask.values())))


def total_params(m = None):
    return int(sum((lambda .0: for v in .0:
v.numel())(m.values())))


def compute_jaccard(mask_a = None, mask_b = None, keys = None):
    if not keys:
        pass
    keys = align_keys(mask_a, mask_b)
    inter = union = 0
    kept_a = kept_b = 0
    for k in keys:
        a = mask_a[k].bool()
        b = mask_b[k].bool()
        kept_a += int(a.sum())
        kept_b += int(b.sum())
        inter += int((a & b).sum())
        union += int((a | b).sum())
    jacc = inter / union if union else 0
    if kept_b:
        return {
            'intersection': inter,
            'union': union,
            'jaccard': jacc,
            'kept_a': kept_a,
            'kept_b': kept_b,
            'a_only': kept_a - inter,
            'b_only': kept_b - inter,
            'prec_a_in_b': inter / kept_a if kept_a else 0,
            'prec_b_in_a': inter / kept_b }
    return {
        'intersection': None,
        'union': inter,
        'jaccard': union,
        'kept_a': jacc,
        'kept_b': kept_a,
        'a_only': kept_b,
        'b_only': kept_a - inter,
        'prec_a_in_b': kept_b - inter,
        'prec_b_in_a': inter / kept_a if kept_a else 0 }


def get_topk_mask(scores = None, target_budget = None, keys = None):
    '''Global top-k bool mask from per-weight scores (exact kept count when k < total).'''
    if not keys:
        pass
    keys = sorted(scores.keys())
    subset = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)
    flat = None((lambda .0 = None: [ subset[k].flatten() for k in .0 ])(keys))
    total = flat.numel()
    k = min(max(0, target_budget), total)
    if k == 0:
        return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)
    if None >= total:
        return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)
    thresh = None.kthvalue(flat, (total - k) + 1).values
    return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)


def module_dispute_prior(name = None, dispute_bias = None):
    '''Heuristic tie-break when float saliency is unavailable.'''
    comp = component(name)
    if dispute_bias == 'tdso':
        if comp in ATTN_SUFFIXES:
            return 2
        if None in MLP_SUFFIXES:
            return 0.5
        return None
    if None in MLP_SUFFIXES:
        return 2
    if None in ATTN_SUFFIXES:
        return 0.5


def build_blended_dispute_scores(m_tacq, m_tdso, tacq_scores = None, tdso_scores = None, lambda_bias = None, keys = ('m_tacq', 'dict[str, torch.Tensor]', 'm_tdso', 'dict[str, torch.Tensor]', 'tacq_scores', 'dict[str, torch.Tensor]', 'tdso_scores', 'dict[str, torch.Tensor]', 'lambda_bias', 'float', 'keys', 'list[str]', 'return', 'dict[str, torch.Tensor]')):
    '''Continuous dispute blend on XOR: λ·T-DSO + (1−λ)·TaCQ (per-tensor max-normalized).'''
    lam = float(max(0, min(1, lambda_bias)))
    out = { }
    for k in keys:
        xor = m_tacq[k].bool() ^ m_tdso[k].bool()
        ts = tdso_scores[k].float().abs()
        cs = tacq_scores[k].float().abs()
        t_norm = ts / (ts.max() + 1e-08)
        c_norm = cs / (cs.max() + 1e-08)
        blend = lam * t_norm + (1 - lam) * c_norm
        out[k] = blend * xor.float()
    return out


def build_dispute_scores(m_tacq, m_tdso, tacq_scores = None, tdso_scores = None, dispute_bias = None, keys = ('m_tacq', 'dict[str, torch.Tensor]', 'm_tdso', 'dict[str, torch.Tensor]', 'tacq_scores', 'dict[str, torch.Tensor] | None', 'tdso_scores', 'dict[str, torch.Tensor] | None', 'dispute_bias', 'str', 'keys', 'list[str]', 'return', 'dict[str, torch.Tensor]')):
    '''Scores on XOR region only, for filling post-core budget.'''
    out = { }
    for k in keys:
        a = m_tacq[k].bool()
        b = m_tdso[k].bool()
        xor = a ^ b
        if not int(xor.sum()):
            out[k] = torch.zeros_like(a, torch.float32, **('dtype',))
            continue
        base = torch.zeros_like(a, torch.float32, **('dtype',))
        tdso_only = ~a & b
        tacq_only = a & ~b
        if dispute_bias == 'tdso':
            side = tdso_only
            side_scores = tdso_scores[k] if tdso_scores else None
        else:
            side = tacq_only
            side_scores = tacq_scores[k] if tacq_scores else None
        if side_scores is not None:
            ranked = side_scores.float().abs() * side.float()
        else:
            pri = module_dispute_prior(k, dispute_bias)
            ranked = pri * side.float()
        out[k] = ranked * xor.float()
    return out


def build_anchored_union_mask(m_tacq, m_tdso, target_budget, dispute_bias, tacq_scores = None, tdso_scores = None, keys = None, lambda_bias = ('tdso', None, None, None, None, None), custom_dispute_scores = ('m_tacq', 'dict[str, torch.Tensor]', 'm_tdso', 'dict[str, torch.Tensor]', 'target_budget', 'int', 'dispute_bias', 'str', 'tacq_scores', 'dict[str, torch.Tensor] | None', 'tdso_scores', 'dict[str, torch.Tensor] | None', 'keys', 'list[str] | None', 'lambda_bias', 'float | None', 'custom_dispute_scores', 'dict[str, torch.Tensor] | None', 'return', 'tuple[dict[str, torch.Tensor], dict[str, Any]]')):
    '''v3-e: lock M_core = M_TaCQ ∩ M_TDSO; fill remainder from XOR via dispute scores.'''
    if not keys:
        pass
    keys = align_keys(m_tacq, m_tdso)
    anchored = { }
    core_count = 0
    for k in keys:
        core = m_tacq[k].bool() & m_tdso[k].bool()
        anchored[k] = core.clone()
        core_count += int(core.sum())
    remaining = target_budget - core_count
    meta = {
        'variant': 'v3-e_anchored_union',
        'dispute_bias': dispute_bias,
        'lambda_bias': lambda_bias,
        'target_budget': target_budget,
        'core_count': core_count,
        'core_fraction_of_budget': core_count / target_budget if target_budget else 0,
        'remaining_budget': max(0, remaining) }
    if remaining <= 0:
        meta['dispute_added'] = 0
        meta['final_kept'] = core_count
        return (anchored, meta)
    if None is not None:
        dispute_scores = custom_dispute_scores
    elif lambda_bias is not None and tacq_scores is not None and tdso_scores is not None:
        dispute_scores = build_blended_dispute_scores(m_tacq, m_tdso, tacq_scores, tdso_scores, lambda_bias, keys)
        meta['dispute_bias'] = f'''blend_lambda={lambda_bias:.3f}'''
    else:
        dispute_scores = build_dispute_scores(m_tacq, m_tdso, tacq_scores, tdso_scores, dispute_bias, keys)
    dispute_mask = get_topk_mask(dispute_scores, remaining, keys, **('keys',))
    dispute_added = 0
    for k in keys:
        before = int(anchored[k].sum())
        anchored[k] = anchored[k] | dispute_mask[k]
        dispute_added += int(anchored[k].sum()) - before
    meta['dispute_added'] = dispute_added
    meta['final_kept'] = count_kept(anchored)
    return (anchored, meta)


def build_module_split_mask(tacq_scores = None, tdso_scores = None, target_budget = None, attn_fraction = (0.3, None), keys = ('tacq_scores', 'dict[str, torch.Tensor]', 'tdso_scores', 'dict[str, torch.Tensor]', 'target_budget', 'int', 'attn_fraction', 'float', 'keys', 'list[str] | None', 'return', 'tuple[dict[str, torch.Tensor], dict[str, Any]]')):
    '''v3-c: attn slots by T-DSO scores, MLP slots by TaCQ scores.'''
    if not keys:
        pass
    keys = align_keys(tacq_scores, tdso_scores)
    k_attn = max(1, int(round(target_budget * attn_fraction)))
    k_mlp = max(1, target_budget - k_attn)
    attn_scores = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)
    mlp_scores = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(keys)
    attn_mask = get_topk_mask(attn_scores, k_attn) if attn_scores else { }
    mlp_mask = get_topk_mask(mlp_scores, k_mlp) if mlp_scores else { }
    out = { }
    for k in keys:
        m = torch.zeros_like(tacq_scores[k], torch.bool, **('dtype',))
        if k in attn_mask:
            m = m | attn_mask[k]
        if k in mlp_mask:
            m = m | mlp_mask[k]
        out[k] = m
    kept = count_kept(out)
    meta = {
        'variant': 'v3-c_module_split',
        'target_budget': target_budget,
        'attn_fraction': attn_fraction,
        'k_attn_target': k_attn,
        'k_mlp_target': k_mlp,
        'final_kept': kept }
    return (out, meta)


def layer_module_heatmap(mask_a = None, mask_b = None, label_a = None, label_b = ('A', 'B', 32), num_layers = ('mask_a', 'dict[str, torch.Tensor]', 'mask_b', 'dict[str, torch.Tensor]', 'label_a', 'str', 'label_b', 'str', 'num_layers', 'int', 'return', 'dict[str, Any]')):
    '''Layer × module disagreement counts for plotting.'''
    keys = align_keys(mask_a, mask_b)
    by_layer = defaultdict((lambda : defaultdict((lambda : {
'inter': 0,
'a_only': 0,
'b_only': 0 }))
))
    by_comp = defaultdict((lambda : {
'inter': 0,
'a_only': 0,
'b_only': 0 }))
    for k in keys:
        a = mask_a[k].bool()
        b = mask_b[k].bool()
        i = int((a & b).sum())
        ao = int((a & ~b).sum())
        bo = int((~a & b).sum())
        li = layer_idx(k)
        comp = component(k)
        by_layer[li][comp]['inter'] += i
        by_layer[li][comp]['a_only'] += ao
        by_layer[li][comp]['b_only'] += bo
        by_comp[comp]['inter'] += i
        by_comp[comp]['a_only'] += ao
        by_comp[comp]['b_only'] += bo
    layer_rows = []
    for li in range(num_layers):
        if li not in by_layer:
            continue
        comps = dict(by_layer[li])
        tot_i = sum((lambda .0: for c in .0:
c['inter'])(comps.values()))
        tot_ao = sum((lambda .0: for c in .0:
c['a_only'])(comps.values()))
        tot_bo = sum((lambda .0: for c in .0:
c['b_only'])(comps.values()))
        layer_rows.append({
            'layer': li,
            'inter': tot_i,
            'a_only': tot_ao,
            'b_only': tot_bo,
            'exclusive_total': tot_ao + tot_bo,
            'components': comps })
    layer_rows.sort((lambda r: r['exclusive_total']), True, **('key', 'reverse'))
    return {
        'label_a': label_a,
        'label_b': label_b,
        'global': compute_jaccard(mask_a, mask_b, keys, **('keys',)),
        'by_component': dict(by_comp),
        'by_layer': layer_rows }


def ascii_layer_bar(row = None, width = None):
    ex = row['exclusive_total']
    if ex <= 0:
        return f'''L{row['layer']:2d}  (agreement only)'''
    ao = None['a_only']
    bo = row['b_only']
    a_bar = int(width * ao / ex)
    b_bar = width - a_bar
    return f'''L{row['layer']:2d}  |{'A' * a_bar}{'B' * b_bar}| excl={ex:,}  A_only={ao:,}  B_only={bo:,}'''


def save_mask_bundle(masks = None, out_path = None, meta = None, flat_path = (None,)):
    out_path.parent.mkdir(True, True, **('parents', 'exist_ok'))
    torch.save({
        'masks': masks,
        'meta': meta }, out_path)
    kept = count_kept(masks)
    total = total_params(masks)
    print(f'''[save] {out_path} kept {kept:,}/{total:,} = {100 * kept / total:.4f}%''')
    if flat_path is not None:
        flat = (lambda .0: pass# WARNING: Decompyle incomplete
)(masks.items())
        flat_path.parent.mkdir(True, True, **('parents', 'exist_ok'))
        torch.save(flat, flat_path)
        print(f'''[save] flat GPTQ mask -> {flat_path}''')
        return None


def print_report(heat = None):
    g = heat['global']
    la = heat['label_a']
    lb = heat['label_b']
    print(f'''\n=== global Jaccard ({la} vs {lb}) ===''')
    print(f'''intersection : {g['intersection']:,}''')
    print(f'''union        : {g['union']:,}''')
    print(f'''Jaccard      : {g['jaccard']:.4f}''')
    print(f'''|A∩B|/|A|    : {g['prec_a_in_b']:.4f}''')
    print(f'''|A∩B|/|B|    : {g['prec_b_in_a']:.4f}''')
    print(f'''|A only|     : {g['a_only']:,}''')
    print(f'''|B only|     : {g['b_only']:,}''')
    print('\n=== per component (inter / A_only / B_only) ===')
    for comp in sorted(heat['by_component']):
        c = heat['by_component'][comp]
        print(f'''  {comp:10s}  inter={c['inter']:8,}  {la}_only={c['a_only']:8,}  {lb}_only={c['b_only']:8,}''')
    print('\n=== top divergent layers (exclusive mass) ===')
    for row in heat['by_layer'][:12]:
        print(f'''  {ascii_layer_bar(row)}''')


def resolve_budget(mask_a = None, mask_b = None, target_budget = None, mask_fraction = ('mask_a', 'dict[str, torch.Tensor]', 'mask_b', 'dict[str, torch.Tensor]', 'target_budget', 'int | None', 'mask_fraction', 'float', 'return', 'int')):
    total = total_params(mask_a)
    if target_budget is not None:
        return target_budget
    kept_a = None(mask_a)
    kept_b = count_kept(mask_b)
    if kept_a == kept_b:
        return kept_a
    return None(round(mask_fraction * total))


def parse_args():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(__doc__, **('description',))
    ap.add_argument('--tacq-mask', str(root / 'tacq_data/tacq_msg_spider_mask.pt'), 'TaCQ flat important_mask .pt', **('default', 'help'))
    ap.add_argument('--tdso-mask', str(root / 'tacq_data/tdso_v2_mult_mask.pt'), 'T-DSO v2 mult flat mask .pt', **('default', 'help'))
    ap.add_argument('--tacq-scores', None, 'TaCQ saliency / importances .pt', **('default', 'help'))
    ap.add_argument('--tdso-scores', None, 'T-DSO combined saliency .pt', **('default', 'help'))
    ap.add_argument('--label-a', 'tacq', **('default',))
    ap.add_argument('--label-b', 'tdso_mult', **('default',))
    ap.add_argument('--mask-fraction', float, 0.0035, **('type', 'default'))
    ap.add_argument('--target-budget', int, None, 'FP16 outlier count (default: from masks)', **('type', 'default', 'help'))
    ap.add_argument('--out-dir', str(root / 'tacq_data/results/mask_jaccard'), **('default',))
    ap.add_argument('--build', ('none', 'v3e_tdso', 'v3e_tacq', 'v3c'), 'none', 'Also emit a v3 candidate mask', **('choices', 'default', 'help'))
    ap.add_argument('--attn-fraction', float, 0.3, 'v3-c: fraction of budget for attention (T-DSO ranked)', **('type', 'default', 'help'))
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(True, True, **('parents', 'exist_ok'))
    m_tacq = load_flat_mask(args.tacq_mask)
    m_tdso = load_flat_mask(args.tdso_mask)
    keys = align_keys(m_tacq, m_tdso)
    budget = resolve_budget(m_tacq, m_tdso, args.target_budget, args.mask_fraction)
    tacq_scores = load_scores(args.tacq_scores) if args.tacq_scores else None
    tdso_scores = load_scores(args.tdso_scores) if args.tdso_scores else None
    if args.build == 'v3c':
        if tacq_scores is None or tdso_scores is None:
            raise SystemExit('v3-c requires --tacq-scores and --tdso-scores')
    heat = layer_module_heatmap(m_tacq, m_tdso, args.label_a, args.label_b, **('label_a', 'label_b'))
    heat['inputs'] = {
        'tacq_mask': args.tacq_mask,
        'tdso_mask': args.tdso_mask,
        'target_budget': budget,
        'has_tacq_scores': tacq_scores is not None,
        'has_tdso_scores': tdso_scores is not None }
    print_report(heat)
    heat_path = out_dir / 'jaccard_heatmap.json'
    with heat_path.open('w') as f:
        json.dump(heat, f, 2, **('indent',))
        None(None, None, None)
    with None:
        if not None:
            pass
    print(f'''\n[write] heatmap JSON -> {heat_path}''')
    if args.build == 'none':
        return None
    if None.build == 'v3c':
        (v3_mask, v3_meta) = build_module_split_mask(tacq_scores, tdso_scores, budget, args.attn_fraction, keys, **('attn_fraction', 'keys'))
        v3_meta['source_masks'] = {
            'tacq': args.tacq_mask,
            'tdso': args.tdso_mask }
        out_mask = out_dir / f'''v3c_attn{int(args.attn_fraction * 100)}.pt'''
    elif args.build == 'v3e_tdso':
        pass
    
    bias = 'tacq'
    (v3_mask, v3_meta) = build_anchored_union_mask(m_tacq, m_tdso, budget, bias, tacq_scores, tdso_scores, keys, **('dispute_bias', 'tacq_scores', 'tdso_scores', 'keys'))
    v3_meta['source_masks'] = {
        'tacq': args.tacq_mask,
        'tdso': args.tdso_mask }
    v3_meta['score_sources'] = {
        'tacq_scores': args.tacq_scores,
        'tdso_scores': args.tdso_scores,
        'fallback': 'module_heuristic' if tacq_scores is None else 'saliency' }
    out_mask = out_dir / f'''v3e_{bias}_bias.pt'''
    j_v3_tacq = compute_jaccard(v3_mask, m_tacq, keys, **('keys',))
    j_v3_tdso = compute_jaccard(v3_mask, m_tdso, keys, **('keys',))
    v3_meta['jaccard_vs_tacq'] = j_v3_tacq['jaccard']
    v3_meta['jaccard_vs_tdso'] = j_v3_tdso['jaccard']
    v3_meta['shared_with_tacq'] = j_v3_tacq['prec_a_in_b']
    v3_meta['shared_with_tdso'] = j_v3_tdso['prec_b_in_a']
    save_mask_bundle(v3_mask, out_mask, v3_meta, out_mask.with_suffix('.flat.pt'), **('flat_path',))
    print(f'''[v3] J(v3,tacq)={v3_meta['jaccard_vs_tacq']:.4f}  J(v3,tdso)={v3_meta['jaccard_vs_tdso']:.4f}  core={v3_meta.get('core_count', 'n/a')}''')

if __name__ == '__main__':
    main()
    return None
