"""
Random Forest classification of a LAS, by blocks (memory O(block), whatever the size of the cloud).

The LAS is read once, sequentially. For every block the feature matrix is built with only the
features of the model, the model predicts, and the block is written to the output LAS as it was
(raw record bytes: coordinates, POINT_ID, every attribute untouched) plus the `prediction` Extra Byte
(uint8). The order of the records is preserved, so the output can be merged back into the point cloud
by pc_columns.py without a join.

The output stays a LAS (and not only a column of the point cloud) because it is a deliverable
(the ZIP package of the viewer) and the input of the pc_classified/ fallback.
"""
import argparse
import os
import pickle
import struct
import sys
import time

import numpy as np

try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cuRF = None
    GPU_AVAILABLE = False

import pipeline_common as pcm

PRED_NAME = 'prediction'


def read_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if 'cuml' in str(e) or 'cuml' in getattr(e, 'name', ''):
            raise RuntimeError('Failed to unpickle model — cuML objects require RAPIDS/cuML.') from e
        raise


def feature_names_in_las(las):
    """Features a model can use: the geometry (offset-subtracted), intensity, colour and the Extra Bytes."""
    return ['X', 'Y', 'Z', 'intensity'] + [c for c in ('red', 'green', 'blue') if c in las.fields] + \
        [n for n in las.extra_names]


build_features = pcm.build_features


def output_header(las):
    """
    Header + VLRs of the output: the input ones with the `prediction` Extra Byte added (or reused when
    the LAS already has a uint8 one). Returns (bytes, offset of the prediction inside the record).
    """
    if las.evlr_count:
        raise ValueError('LAS with Extended VLRs are not supported')
    hb = bytearray(las.header_bytes)
    if PRED_NAME in las.extra_names:
        dt, off = las.fields[PRED_NAME]
        if np.dtype(dt) != np.dtype('<u1'):
            raise ValueError("the LAS already has a 'prediction' attribute that is not uint8")
        return bytes(hb), off
    descr = struct.pack('<2sBB32s4s24s24s24s24s24s32s', b'\0\0', 1, 0, PRED_NAME.encode(), b'\0' * 4,
                        b'\0' * 24, b'\0' * 24, b'\0' * 24, b'\0' * 24, b'\0' * 24, b'prediction (uint8)'.ljust(32, b'\0'))
    assert len(descr) == 192
    if las.extra_vlr is not None:
        vpos, length = las.extra_vlr
        if length + 192 > 65535:
            raise ValueError('Extra Bytes VLR is full')
        end = vpos + 54 + length
        struct.pack_into('<H', hb, vpos + 20, length + 192)
        hb[end:end] = descr
        added = 192
    else:
        vlr = struct.pack('<H16sHH32s', 0, b'LASF_Spec', 4, 192, b'Extra bytes'.ljust(32, b'\0')) + descr
        struct.pack_into('<I', hb, 100, las.num_vlrs + 1)
        hb[las.vlr_end:las.vlr_end] = vlr
        added = len(vlr)
    struct.pack_into('<I', hb, 96, las.data_offset + added)
    struct.pack_into('<H', hb, 105, las.record_len + 1)
    return bytes(hb), las.record_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_features', nargs="+", required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--test_filepath', required=True)
    parser.add_argument('--output_classify_name', required=True)
    parser.add_argument('--use_gpu', action='store_true')
    pcm.add_common_args(parser)
    args = parser.parse_args()

    t0 = time.time()
    budget = pcm.memory_budget_bytes(args.memory_budget)
    print('\nLoading model ...', flush=True)
    model = read_model(args.model)

    print('Opening testing data ...', flush=True)
    las = pcm.LasFile(args.test_filepath)
    available = feature_names_in_las(las)
    features = []
    for feat in args.selected_features:
        if feat in available:
            features.append(feat)
        else:
            print(f"Warning: feature '{feat}' not found in header", flush=True)
    print(f"Header: {available + [PRED_NAME]}", flush=True)
    print(f"\nTesting samples: {las.n}", flush=True)
    print(f"Using features: {args.selected_features}", flush=True)

    n_classes = len(getattr(model, 'classes_', [])) or 8
    nf = len(features)
    rows = int(max(1 << 16, min(4 << 20, budget * 0.4 / (3 * (las.record_len + 1) + 8 * nf + 16 * n_classes + 64))))
    gpu = args.use_gpu and GPU_AVAILABLE and cuRF is not None and isinstance(model, cuRF)
    if gpu:
        print("GPU requested and it is AVAILABLE, so use cuml for classification.", flush=True)
    elif args.use_gpu and not GPU_AVAILABLE:
        print('Warning: --use_gpu requested but cuML not available; using CPU.', flush=True)
    else:
        print("GPU is NOT requested, so use scikit-learn for classification; using CPU.", flush=True)
    print(f"Block: {rows} points (memory budget {budget // 1048576} MB)", flush=True)

    out_dir = os.path.dirname(args.output_classify_name)
    if out_dir and not os.path.exists(out_dir):
        print(f"Creating output directory: {out_dir}", flush=True)
        os.makedirs(out_dir, exist_ok=True)
    tmp_out = args.output_classify_name + '.tmp'
    pcm.install_term_cleanup(lambda: [tmp_out])

    header, pred_off = output_header(las)
    reuse = PRED_NAME in las.extra_names
    names = [f for f in dict.fromkeys(features)]
    dt = las.block_dtype(names)
    X = np.empty((rows, nf), dtype=np.float32)
    counts = {}
    tm = {'wait_read': 0.0, 'features': 0.0, 'predict': 0.0, 'write': 0.0}
    t1 = time.time()
    print(f'---> Loading time {int((t1 - t0) // 60)} min {int((t1 - t0) % 60)} sec', flush=True)
    print('\nClassifying ...', flush=True)
    try:
        with pcm.SequentialWriter(tmp_out) as w:
            w.write(header)
            done = 0
            last_pct = -1
            tw = time.time()
            for start, buf in pcm.prefetched(las.raw_blocks(rows)):
                tm['wait_read'] += time.time() - tw
                t_a = time.time()
                m = buf.shape[0]
                blk = np.ndarray(shape=(m,), dtype=dt, buffer=buf)
                Xb = X[:m]
                build_features(blk, features, las, Xb)
                t_b = time.time()
                if gpu:
                    y = cp.asnumpy(model.predict(cp.asarray(Xb)))
                else:
                    y = model.predict(Xb)
                t_c = time.time()
                tm['features'] += t_b - t_a
                tm['predict'] += t_c - t_b
                y = np.asarray(y).reshape(-1).astype(np.uint8)
                for v, c in zip(*np.unique(y, return_counts=True)):
                    counts[int(v)] = counts.get(int(v), 0) + int(c)
                if reuse:
                    buf[:, pred_off] = y
                    w.write(buf.tobytes())
                else:
                    out = np.empty((m, las.record_len + 1), dtype=np.uint8)
                    out[:, :las.record_len] = buf
                    out[:, las.record_len] = y
                    w.write(out.tobytes())
                tm['write'] += time.time() - t_c
                done += m
                pct = int(100 * done / las.n)
                if pct != last_pct and pct % 5 == 0:
                    print(f'[progress] classify {pct}%', flush=True)
                    last_pct = pct
                tw = time.time()
            # anything after the points of the input (never written by the pipeline) is not copied
        os.replace(tmp_out, args.output_classify_name)
    except BaseException:
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        raise

    t2 = time.time()
    print('  timings (s): ' + ', '.join(f'{k} {v:.1f}' for k, v in tm.items()), flush=True)
    print(f'---> Classification time {int((t2 - t1) // 60)} min {int((t2 - t1) % 60)} sec', flush=True)
    print(f'  Written {las.n} points → {args.output_classify_name}  (classes: {dict(sorted(counts.items()))})', flush=True)
    print(f'\nTotal time {int((t2 - t0) // 60)} min {int((t2 - t0) % 60)} sec\n', flush=True)


if __name__ == '__main__':
    main()
