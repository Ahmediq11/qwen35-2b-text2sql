"""Apply / verify / hash weight edits on safetensors checkpoints (numpy only, no torch).

Reimplemented after reviewing ref/lm_understand-logs/exp8_math/code/{st_numpy,apply_edit}.py.
BF16 tensors are handled as raw uint16 bit patterns so hashes match the reference manifests bit for bit.

    python src/edits.py verify <base.safetensors> <edit_dir>              # rebuild + check manifest hashes
    python src/edits.py compare <a.safetensors> <b.safetensors> <name>...  # sha256 of the named tensors in two files
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64,
          "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load(path, names=None):
    """{name: (dtype string, array)}; BF16 arrays are uint16 bit patterns."""
    head, off = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, info in head.items():
            if name == "__metadata__" or (names is not None and name not in names):
                continue
            a, b = info["data_offsets"]
            f.seek(off + a)
            out[name] = (info["dtype"],
                         np.frombuffer(f.read(b - a), dtype=DTYPES[info["dtype"]]).reshape(info["shape"]))
    return out


def save(path, tensors, metadata=None):
    head, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, arr = tensors[name]
        b = np.ascontiguousarray(arr).tobytes()
        head[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if metadata:
        head["__metadata__"] = metadata
    h = json.dumps(head, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def bf16_bits(x32):
    """float32 -> bfloat16 bits, round to nearest even (matches torch for finite values)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def rebuild(base_bits, entry, z, i):
    """Rebuild one edited tensor from the reference diff format (sparse entries or low-rank factors + fixups)."""
    if entry["form"] == "lowrank":
        w = bf16_to_f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        e = bf16_bits(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    return e.reshape(base_bits.shape)


def apply_edit(base_path, edit_dir):
    """Returns ({name: edited bits}, report rows); raises if any base or edited hash differs from the manifest."""
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
    names = [t["name"] for t in man["tensors"]]
    base = load(base_path, names)
    out, rows = {}, []
    for i, t in enumerate(man["tensors"]):
        dt, b = base[t["name"]]
        assert dt == "BF16" and list(b.shape) == t["shape"], (t["name"], dt, b.shape)
        hb = sha256(b)
        e = rebuild(b, t, z, i)
        he = sha256(e)
        rows.append({"name": t["name"], "base_ok": hb == t["base_sha256"], "edited_ok": he == t["edited_sha256"],
                     "changed_entries": int((e != b).sum()), "changed_columns": int((e != b).any(0).sum()),
                     "edited_sha256": he})
        out[t["name"]] = e
    return out, rows


def main(argv):
    if argv[0] == "verify":
        _, rows = apply_edit(argv[1], argv[2])
        print(json.dumps(rows, indent=1))
        sys.exit(0 if all(r["base_ok"] and r["edited_ok"] for r in rows) else 1)
    if argv[0] == "compare":
        a, b = load(argv[1], argv[3:]), load(argv[2], argv[3:])
        for n in argv[3:]:
            print(n, sha256(a[n][1]), sha256(b[n][1]))


if __name__ == "__main__":
    main(sys.argv[1:])
