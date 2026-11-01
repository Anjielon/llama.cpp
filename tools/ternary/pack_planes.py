#!/usr/bin/env python3
"""Pack separate ternary plane tensors into one multi-plane tensor.

Some ternary models store extra planes of a weight as separate tensors that a modified graph sums:
    blk.N.attn_qkv.weight, blk.N.attn_qkv2.weight, blk.N.attn_qkv3.weight  ->  W = W1 + W2 + W3
This tool rewrites them as ONE tensor of a packed type (TQ1_0R_1024_X{2,3,4} or TQ1_0_X{2,3,4}):
a block holds the K planes of the same weights back to back, so the model graph is the standard one
and the packed tensor costs exactly the bytes of the K plane tensors (byte-for-byte the same data).

Packed: every family whose planes 1..K (K >= 2) share base type (TQ1_0R_1024 or TQ1_0) and shape.
Left as they are: routed-expert planes (ffn_*_exps2 + map cover a subset of experts, not a plane of the
whole tensor), and anything not matching the rule above (reported).

    python3 tools/ternary/pack_planes.py IN.gguf OUT.gguf [--check]

--check re-reads OUT and verifies that every packed tensor unpacks to the original plane bytes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402

PACKED = {
    QT.TQ1_0R_1024: {2: QT.TQ1_0R_1024_X2, 3: QT.TQ1_0R_1024_X3, 4: QT.TQ1_0R_1024_X4},
    QT.TQ1_0: {2: QT.TQ1_0_X2, 3: QT.TQ1_0_X3, 4: QT.TQ1_0_X4},
}
PLANE_RE = re.compile(r'^(?P<stem>.+?)(?P<k>[234])(?P<suffix>\.weight)$')


def plan(reader: gguf.GGUFReader) -> tuple[dict[str, list], set[str], list[str]]:
    """base name -> [plane tensors 1..K]; names absorbed into a pack; notes on families left alone."""
    by_name = {t.name: t for t in reader.tensors}
    groups: dict[str, list] = {}
    absorbed: set[str] = set()
    notes: list[str] = []
    for t in reader.tensors:
        m = PLANE_RE.match(t.name)
        if not m or m.group('k') != '2':
            continue
        base_name = m.group('stem') + m.group('suffix')
        base = by_name.get(base_name)
        if base is None or '_exps' in base_name:
            continue
        planes = [base]
        k = 2
        while (m.group('stem') + str(k) + m.group('suffix')) in by_name:
            planes.append(by_name[m.group('stem') + str(k) + m.group('suffix')])
            k += 1
        ty = base.tensor_type
        if ty not in PACKED or len(planes) > 4:
            notes.append(f'{base_name}: {len(planes)} planes of {ty.name}, not packed')
            continue
        if any(p.tensor_type != ty or list(p.shape) != list(base.shape) for p in planes):
            notes.append(f'{base_name}: planes differ in type/shape, not packed')
            continue
        groups[base_name] = planes
        absorbed.update(p.name for p in planes[1:])
    return groups, absorbed, notes


def pack_bytes(planes: list) -> np.ndarray:
    """Interleave the K planes block by block: out[b] = plane1[b] | plane2[b] | ... ; shape (rows, K*row_bytes)."""
    ty = planes[0].tensor_type
    _, bsz = gguf.GGML_QUANT_SIZES[ty]
    rows = planes[0].data.shape[0] if planes[0].data.ndim > 1 else 1
    flat = [np.ascontiguousarray(p.data).reshape(-1, bsz) for p in planes]
    out = np.stack(flat, axis=1)  # (n_blocks, K, bsz)
    return out.reshape(rows, -1)


def copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)


def convert(src: Path, dst: Path, drop: bool = False) -> dict:
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    groups, absorbed, notes = plan(reader)
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    copy_kv(reader, writer)
    out = []
    for t in reader.tensors:
        if t.name in absorbed:
            continue
        if t.name in groups and drop:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            out.append((t.name, t.tensor_type, [t]))
        elif t.name in groups:
            planes = groups[t.name]
            ty = PACKED[t.tensor_type][len(planes)]
            out.append((t.name, ty, planes))
            data_shape = (t.data.shape[0] if t.data.ndim > 1 else 1, t.data.shape[-1] * len(planes))
            if t.data.ndim > 2:
                raise ValueError(f'{t.name}: packing of >2D tensors not supported')
            writer.add_tensor_info(t.name, data_shape, np.dtype(np.uint8), int(t.n_bytes) * len(planes), ty)
        else:
            out.append((t.name, t.tensor_type, [t]))
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, ty, planes in out:
        data = pack_bytes(planes) if len(planes) > 1 else planes[0].data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    return {'packed': len(groups), 'absorbed': len(absorbed), 'notes': notes,
            'kinds': sorted({f'{PACKED[g[0].tensor_type][len(g)].name}' for g in groups.values()})}


def check(src: Path, dst: Path) -> int:
    a = gguf.GGUFReader(src)
    b = gguf.GGUFReader(dst)
    by_a = {t.name: t for t in a.tensors}
    groups, absorbed, _ = plan(a)
    bad = 0
    for t in b.tensors:
        if t.name in groups:
            planes = groups[t.name]
            _, bsz = gguf.GGML_QUANT_SIZES[planes[0].tensor_type]
            got = np.ascontiguousarray(t.data).reshape(-1, len(planes), bsz)
            for k, p in enumerate(planes):
                if not np.array_equal(got[:, k, :], np.ascontiguousarray(p.data).reshape(-1, bsz)):
                    print(f'MISMATCH {t.name} plane {k + 1}')
                    bad += 1
        elif not np.array_equal(np.ascontiguousarray(t.data), np.ascontiguousarray(by_a[t.name].data)):
            print(f'MISMATCH {t.name}')
            bad += 1
    n_a = sum(int(t.n_bytes) for t in a.tensors)
    n_b = sum(int(t.n_bytes) for t in b.tensors)
    print(f'check: {len(groups)} packed, {bad} mismatches, tensor bytes {n_a} -> {n_b}')
    return bad + (n_a != n_b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--drop-extra-planes', action='store_true',
                    help='ablation: keep only plane 1 of every packable family (negative control for tests)')
    a = ap.parse_args()
    r = convert(a.src, a.dst, drop=a.drop_extra_planes)
    if a.drop_extra_planes:
        print(f"dropped the extra planes of {r['packed']} families ({r['absorbed']} tensors)")
        return 0
    print(f"packed {r['packed']} families ({r['absorbed']} plane tensors absorbed): {', '.join(r['kinds'])}")
    for n in r['notes']:
        print('  note:', n)
    return check(a.src, a.dst) if a.check else 0


if __name__ == '__main__':
    sys.exit(main())
