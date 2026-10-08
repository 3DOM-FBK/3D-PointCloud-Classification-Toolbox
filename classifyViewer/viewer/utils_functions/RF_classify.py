import time
import pickle
import argparse
import numpy as np
import struct
try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cuRF = None
    GPU_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier
import laspy
import re
import os

# ─────────────────────────────────────────────────────────────────────────────
#  Raw VLR reader — reads Extra Bytes dim names directly from binary
# ─────────────────────────────────────────────────────────────────────────────

def _read_vlr_extra_names(filepath):
    result = []
    with open(filepath, 'rb') as f:
        f.seek(94);  hdr_size, = struct.unpack('<H', f.read(2))
        f.seek(100); num_vlrs, = struct.unpack('<I', f.read(4))
        f.seek(hdr_size)
        for _ in range(num_vlrs):
            hdr = f.read(54)
            if len(hdr) < 54: break
            user_id   = hdr[2:18].rstrip(b'\x00').decode('latin-1')
            record_id = struct.unpack_from('<H', hdr, 18)[0]
            rec_len   = struct.unpack_from('<H', hdr, 20)[0]
            if user_id == 'LASF_Spec' and record_id == 4:
                data = f.read(rec_len)
                for i in range(rec_len // 192):
                    rec      = data[i*192:(i+1)*192]
                    dtype    = rec[2]
                    raw_name = rec[4:36].rstrip(b'\x00')
                    name     = raw_name.decode('latin-1').strip()
                    if name: result.append((name, dtype))
                return result
            else:
                f.seek(rec_len, 1)
    return result

# ─────────────────────────────────────────────────────────────────────────────
#  Data I/O
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_indices(full_header, selected_features):
    indices = []
    for feat in selected_features:
        if feat in full_header:
            indices.append(full_header.index(feat))
        else:
            print(f"Warning: feature '{feat}' not found in header")
    return indices

def read_model(filepath):
    try:
        return pickle.load(open(filepath, 'rb'))
    except Exception as e:
        if 'cuml' in str(e) or 'cuml' in getattr(e, 'name', ''):
            raise RuntimeError('Failed to unpickle model — cuML objects require RAPIDS/cuML.') from e
        raise

def read_las_data(filepath):
    las = laspy.read(filepath)
    # SALVATAGGIO OFFSET E SCALE ORIGINALI
    original_metadata = {
        'offsets': np.array(las.header.offsets), 
        'scales': np.array(las.header.scales),
        # Keep full-precision world coordinates to avoid XY drift on rewrite
        'world_x': np.array(las.x, dtype=np.float64),
        'world_y': np.array(las.y, dtype=np.float64),
        'world_z': np.array(las.z, dtype=np.float64)
    }
    
    columns = []
    header = []
    all_dims = las.point_format.dimension_names
    
    technical_metadata = {
        'return_number', 'number_of_returns', 'scan_direction_flag',
        'edge_of_flight_line', 'classification', 'synthetic', 
        'key_point', 'withheld', 'scan_angle_rank', 'scan_angle', 
        'user_data', 'point_source_id', 'gps_time', 'scanner_channel', 'overlap'
    }
    
    # ESTRAZIONE COORDINATE SCALATE E SOTTRAZIONE OFFSET
    raw_x = np.array(las.x, dtype=np.float64)
    raw_y = np.array(las.y, dtype=np.float64)
    raw_z = np.array(las.z, dtype=np.float64)

    for dim in all_dims:
        dim_low = dim.lower()
        if dim == 'X':
            columns.append(raw_x - original_metadata['offsets'][0])
            header.append(dim)
        elif dim == 'Y':
            columns.append(raw_y - original_metadata['offsets'][1])
            header.append(dim)
        elif dim == 'Z':
            columns.append(raw_z - original_metadata['offsets'][2])
            header.append(dim)
        elif dim_low not in technical_metadata:
            try:
                col_data = np.array(getattr(las, dim), dtype=np.float64)
                columns.append(col_data)
                header.append(dim)
            except AttributeError:
                continue

    # POINT_ID goes through a float32 matrix (exact only up to 2^24): keep the real uint32 values aside
    # and use them when writing, otherwise ids above 16 777 216 are corrupted.
    for dim in list(las.point_format.dimension_names):  # all_dims is a one-shot generator
        if dim.lower() in ('point_id', 'pointid'):
            original_metadata['point_id'] = np.array(las[dim], dtype=np.uint32)
            break

    if 'prediction' not in header:
        header.append('prediction')

    X = np.column_stack(columns)
    print(f"Header: {header}")
    return np.asarray(X, dtype=np.float32), header, original_metadata

def write_classification_las(X, Y, filename, header, original_metadata=None):
    X = np.array(X)
    Y = np.array(Y).flatten()
    n_points = len(X)

    LAS_NATIVE = {
        'x', 'y', 'z', 'intensity', 'return_number', 'number_of_returns',
        'scan_direction_flag', 'edge_of_flight_line', 'classification',
        'synthetic_flag', 'keypoint_flag', 'withheld_flag', 'scan_angle_rank',
        'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue'
    }

    las_header = laspy.LasHeader(point_format=3, version="1.2")

    # APPLICAZIONE OFFSET E SCALE ORIGINALI NELL'HEADER
    if original_metadata is not None:
        las_header.offsets = original_metadata['offsets']
        las_header.scales = original_metadata['scales']
    
    extra_dim_defs = []
    for col_name in header:
        cl = col_name.lower()
        if cl in LAS_NATIVE or cl == 'prediction':
            continue
        dtype = np.uint32 if 'point_id' in cl or 'pointid' in cl else np.float32
        extra_dim_defs.append(laspy.ExtraBytesParams(name=col_name, type=dtype))

    extra_dim_defs.append(laspy.ExtraBytesParams(name='prediction', type=np.uint8))
    las_header.add_extra_dims(extra_dim_defs)
    las = laspy.LasData(header=las_header)

    # INDICI PER X, Y, Z
    xi, yi, zi = header.index('X'), header.index('Y'), header.index('Z')

    # RIPRISTINO COORDINATE GLOBALI (X + Offset) PRIMA DEL SALVATAGGIO
    if original_metadata is not None and all(k in original_metadata for k in ('world_x', 'world_y', 'world_z')):
        las.x = original_metadata['world_x']
        las.y = original_metadata['world_y']
        las.z = original_metadata['world_z']
    else:
        las.x = X[:, xi] + original_metadata['offsets'][0]
        las.y = X[:, yi] + original_metadata['offsets'][1]
        las.z = X[:, zi] + original_metadata['offsets'][2]

    for i, col_name in enumerate(header):
        cl = col_name.lower()
        if cl in ['x', 'y', 'z', 'prediction']: continue
        
        data = X[:, i]
        if cl == 'red': las.red = data.astype(np.uint16)
        elif cl == 'green': las.green = data.astype(np.uint16)
        elif cl == 'blue': las.blue = data.astype(np.uint16)
        elif cl == 'intensity': las.intensity = data.astype(np.uint16)
        elif ('point_id' in cl or 'pointid' in cl) and original_metadata is not None and 'point_id' in original_metadata:
            las[col_name] = original_metadata['point_id']
        elif cl == 'scan_angle': las.scan_angle_rank = np.clip(np.rint(data), -128, 127).astype(np.int8)
        else:
            las[col_name] = data.astype(np.float32 if 'point_id' not in cl else np.uint32)

    las['prediction'] = Y.astype(np.uint8)    
    las.write(filename)
    print(f"  Written {n_points} points → {filename}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_features', nargs="+", required=True)
    parser.add_argument('--model',             required=True)
    parser.add_argument('--test_filepath',     required=True)
    parser.add_argument('--output_classify_name', required=True)
    parser.add_argument('--use_gpu',           action='store_true')
    args = parser.parse_args()

    t0 = time.time()
    print('\nLoading model ...')
    model = read_model(args.model)

    print('Loading testing data ...')
    X, header, original_metadata = read_las_data(args.test_filepath)

    feat_to_use = get_feature_indices(header, args.selected_features)

    t1 = time.time()
    print(f'---> Loading time {int((t1-t0)//60)} min {int((t1-t0)%60)} sec')
    print(f"\nTesting samples: {len(X)}")
    print(f"Using features: {args.selected_features}")

    print('\nClassifying ...')
    if args.use_gpu and GPU_AVAILABLE and cuRF is not None and isinstance(model, cuRF):
        print(f"GPU requested and it is AVAILABLE, so use cuml for training.")
        Y_pred = cp.asnumpy(model.predict(cp.asarray(X[:, feat_to_use])))
    else:
        if args.use_gpu and not GPU_AVAILABLE:
            print('Warning: --use_gpu requested but cuML not available; using CPU.')
        elif not args.use_gpu:
            print(f"GPU is NOT requested, so use scikit-learn for training; using CPU.")
        Y_pred = model.predict(X[:, feat_to_use])

    t2 = time.time()
    print(f'---> Classification time {int((t2-t1)//60)} min {int((t2-t1)%60)} sec')

    print('\nSaving ...')
    # --- CONTROLLO CARTELLA ---
    output_dir = os.path.dirname(args.output_classify_name)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    write_classification_las(X, Y_pred, args.output_classify_name, header, original_metadata)

    t3 = time.time()
    print(f'---> Saving time {int((t3-t2)//60)} min {int((t3-t2)%60)} sec')
    print(f'\nTotal time {int((t3-t0)//60)} min {int((t3-t0)%60)} sec\n')


if __name__ == '__main__':
    main()