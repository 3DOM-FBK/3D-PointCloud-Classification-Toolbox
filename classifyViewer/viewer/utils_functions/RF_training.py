import time
import pickle
import argparse
import os
import sys
import numpy as np
try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cuRF = None
    GPU_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, jaccard_score
import itertools
import laspy
import re
import copy
from scipy.spatial import KDTree

import struct

import pipeline_common as pcm

def _read_vlr_extra_names(filepath):
    """ Legge i nomi reali delle Extra Bytes dal binario LAS """
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
                    rec = data[i*192:(i+1)*192]
                    name = rec[4:36].rstrip(b'\x00').decode('latin-1').strip()
                    if name: result.append(name)
                return result
            else:
                f.seek(rec_len, 1)
    return result

def load_features_and_class(filepath):
    ''' Load the features to use from a txt file. Format: 
        f1 f2 ... fn
        c

        Attributes:
            filepath    :    path to the .txt file containing the features to use
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                feat_to_use = [int(t) for t in tokens]
            elif line_index == 1:
                class_index= int(tokens[0])
    return feat_to_use, class_index

def get_feature_indices(full_header, selected_features):
    ''' Get indices of selected features in the full header.
    
        Attributes:
            full_header        : list of all feature names
            selected_features  : list of feature names to find
        
        Return:
            indices : list of indices of selected features in full_header
    '''
    indices = []
    for feat in selected_features:
        if feat in full_header:
            indices.append(full_header.index(feat))
        else:
            print(f"Warning: feature '{feat}' not found in header")
    return indices

def read_txt_data(filepath, class_index):
    ''' Read a labelled point cloud.

        Attributes:
            filepath    :   path to the .txt file containing the point cloud
            class_index :   index of the class label

        Return:
            X   :   numpy array with features
            Y   :   numpy array with classes
            header :   list of feature names (from the first line)
    '''
    X, Y = [], []
    header = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
        for line_index, line in enumerate(lines):
            tokens = line.strip().split(' ')
            if line_index == 0:
                # Treat first line as header only if it contains non-numeric tokens (actual column names)
                try:
                    [float(t) for t in tokens]
                    # All tokens are numeric — no header present, process as data
                except ValueError:
                    # At least one token is non-numeric — this is a header line
                    header = tokens
                    continue
            if 'nan' not in tokens and not tokens[0].startswith('//') and len(tokens) > class_index:
                X.append([float(t) for t_index, t in enumerate(tokens) if t_index != class_index])
                Y.append(int(float(tokens[class_index])))
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), header

def train_model(X_train, Y_train, n_jobs, use_gpu=False,
                n_estimators=200, max_depth=15, min_samples_split=20, max_features='sqrt'):
    ''' Train the Random Forest model with the specified parameters and return it.

        Default parameters follow the RF200 configuration:
            - 200 trees (n_estimators)
            - maximum tree depth of d_max = 15
            - minimum samples to split a node of n_min = 20
            - number of features per split = sqrt(n_features)

        Attributes:
            X_train             :   numpy array with training features
            Y_train             :   numpy array with training classes
            n_jobs              :   number of threads used to train the model (CPU only)
            use_gpu             :   whether to use GPU-accelerated training via cuML
            n_estimators        :   number of trees in the forest (default: 200)
            max_depth           :   maximum depth of each tree (default: 15)
            min_samples_split   :   minimum samples required to split a node (default: 20)
            max_features        :   number of features per split (default: 'sqrt')
        
        Return:
            model           :   trained model
            features lists:   list of feature importance values for each feature (in the same order as X_train columns)
    '''
    if use_gpu and GPU_AVAILABLE:
        print(f"GPU requested and it is AVAILABLE, so use cuml for training.")
        Xg = cp.asarray(X_train, dtype=cp.float32)
        yg = cp.asarray(Y_train, dtype=cp.int32)
        
        # cuML a volte richiede esplicitamente la precisione delle statistiche
        model = cuRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=0
        )
        model.fit(Xg, yg)
        
        # Se .feature_importances_ fallisce, lo calcoliamo via sklearn su un piccolo subset
        try:
            fi = model.feature_importances_.get()
        except:
            # print("[INFO] cuML non espone importanze. Calcolo stima su subset CPU...")
            subset_idx = np.random.choice(len(X_train), min(100000, len(X_train)), replace=False)
            rf_cpu = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=n_jobs)
            rf_cpu.fit(X_train[subset_idx], Y_train[subset_idx])
            fi = rf_cpu.feature_importances_
        
        return model, fi
   
    else:
        if use_gpu and not GPU_AVAILABLE:
            print('Warning: --use_gpu requested but cuML not available; using CPU.')
        elif not use_gpu:
            print(f"GPU is NOT requested, so use scikit-learn for training; using CPU.")
        # Fallback CPU (scikit-learn)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, max_features=max_features,
            n_jobs=n_jobs, random_state=0
        )
        model.fit(X_train, Y_train)
        return model, model.feature_importances_


def write_classification_txt(X_test, Y_test_pred, filename, header):
    ''' Write the test set with the predicted labels 
    '''
    with open('{}'.format(filename), 'w') as out:
        X_test = X_test.tolist()
        Y_test_pred = Y_test_pred.tolist()
        out.write('//{}\n'.format(header))
        for index, x_t in enumerate(X_test):
            x_t_as_str = " ".join([str(x) for x in x_t[0:6]])
            out.write('{} {}\n'.format(x_t_as_str, str(Y_test_pred[index])))


def available_feature_names(las):
    """Names a model can be trained on: geometry, standard fields and Extra Bytes of the LAS."""
    skip = {'gps_time', 'classification', 'labels'}      # not features: the target and the time (as before)
    names = ['X', 'Y', 'Z'] + [n for n in las.fields if n not in ('X', 'Y', 'Z') and n not in skip]
    return list(dict.fromkeys(names))


def max_training_points_for(n_features, budget_bytes):
    """Points of a labelled set that fit the budget: feature matrix (float32) + one copy made by the fit + labels."""
    return int(max(100000, budget_bytes * 0.5 / (8 * max(1, n_features) + 8)))


def read_las_data(filepath, class_las_index, features, budget_bytes, max_points=None, seed=0, what='data'):
    """
    Reads a labelled LAS by blocks: only the selected `features` (not every dimension) and the label are loaded.
    When the set has more points than `max_points` (default: what the budget allows) it is sampled, stratified by
    class (every class keeps its share, at least min(count, 1000) points), and the sampling is logged.

    Returns X (float32, columns = features found in the LAS, in the order requested), Y (int32), the list of those
    features and the LAS offsets/scales.
    """
    las = pcm.LasFile(filepath)
    avail = available_feature_names(las)
    used = []
    for f in features:
        if f in avail:
            used.append(f)
        else:
            print(f"Warning: feature '{f}' not found in header")
    label = class_las_index if class_las_index in las.fields else 'classification'
    names = list(dict.fromkeys(used + [label]))
    rows = las.block_rows_for(budget_bytes * 0.2, per_row_extra=8 * len(used) + 16)
    cap = int(max_points) if max_points else max_training_points_for(len(used), budget_bytes)

    def labels_of(blk):
        y = np.asarray(blk[label])
        if label == 'classification' and las.fmt <= 5:
            y = y & 0x1F
        return y.astype(np.int32)

    keep_prob = None
    if las.n > cap:
        counts = {}
        for _, blk in las.blocks([label], rows):
            u, c = np.unique(labels_of(blk), return_counts=True)
            for k, v in zip(u, c):
                counts[int(k)] = counts.get(int(k), 0) + int(v)
        total = sum(counts.values())
        floor_n = 1000
        keep_prob = {k: min(1.0, max(cap * v / total, min(v, floor_n)) / v) for k, v in counts.items()}
        expected = sum(keep_prob[k] * v for k, v in counts.items())
        print(f"[Info] {what}: {total} points exceed the limit of {cap} (memory budget "
              f"{budget_bytes // 1048576} MB): stratified sampling by class keeps ~{int(expected)} "
              f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}). Use --max_training_points to change it.",
              flush=True)
    rng = np.random.RandomState(seed)
    Xs, Ys = [], []
    for _, blk in pcm.prefetched(las.blocks(names, rows)):
        y = labels_of(blk)
        if keep_prob is not None:
            lut_keys = np.array(sorted(keep_prob), dtype=np.int64)
            lut_vals = np.array([keep_prob[k] for k in sorted(keep_prob)])
            p = lut_vals[np.searchsorted(lut_keys, y)]
            sel = rng.random_sample(len(y)) < p
        else:
            sel = slice(None)
        Xb = np.empty((len(y), len(used)), dtype=np.float32)
        pcm.build_features(blk, used, las, Xb)
        Xs.append(Xb[sel])
        Ys.append(y[sel])
    X = np.concatenate(Xs) if Xs else np.empty((0, len(used)), dtype=np.float32)
    Y = np.concatenate(Ys) if Ys else np.empty(0, dtype=np.int32)
    meta = {'offsets': np.array(las.offset), 'scales': np.array(las.scale)}
    return X, Y, used, meta

def write_classification_las(X, Y, filename, header, original_metadata):
    xi, yi, zi = header.index('X'), header.index('Y'), header.index('Z')
    
    # Prepariamo l'header
    las_header = laspy.LasHeader(point_format=3, version="1.2")
    las_header.offsets = original_metadata['offsets']
    las_header.scales = original_metadata['scales']

    # Dimensioni standard LAS (non aggiungerle come Extra Bytes)
    LAS_NATIVE = {
        'X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
        'scan_direction_flag', 'edge_of_flight_line', 'classification',
        'synthetic_flag', 'keypoint_flag', 'withheld_flag', 'scan_angle_rank',
        'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue'
    }
    
    # Aggiungiamo Extra Bytes (solo se non sono già nativi)
    for col_name in header:
        if col_name not in LAS_NATIVE:
            try:
                las_header.add_extra_dim(laspy.ExtraBytesParams(name=col_name, type=np.float32))
            except Exception: pass

    las = laspy.LasData(las_header)
    
    # 1. Coordinate (Fondamentali)
    las.x = X[:, xi] + original_metadata['offsets'][0]
    las.y = X[:, yi] + original_metadata['offsets'][1]
    las.z = X[:, zi] + original_metadata['offsets'][2]
    
    # 2. Colori
    for color in ['red', 'green', 'blue']:
        if color in header:
            ci = header.index(color)
            c_data = X[:, ci]
            # Scala a 16-bit se necessario
            val = (c_data * 257) if c_data.max() <= 255 else c_data
            setattr(las, color, val.astype(np.uint16))

    # 3. Altri campi (Iterazione sicura)
    for i, col_name in enumerate(header):
        # Saltiamo quelli già gestiti o problematici
        if col_name in ['X', 'Y', 'Z', 'red', 'green', 'blue', 'classification']:
            continue
            
        try:
            if col_name == 'intensity':
                las.intensity = X[:, i].astype(np.uint16)
            elif col_name == 'gps_time':
                las.gps_time = X[:, i]
            elif col_name == 'scan_angle_rank':
                # Lo convertiamo in intero prima di passarlo per evitare il 'left_shift' error
                las.scan_angle_rank = X[:, i].astype(np.int8)
            elif col_name not in LAS_NATIVE:
                # Questo scrive gli Extra Bytes (Verticality, Sphericity, ecc.)
                las[col_name] = X[:, i].astype(np.float32)
            else:
                # Altri campi nativi (es. user_data)
                setattr(las, col_name, X[:, i].astype(np.int64 if col_name.endswith('time') else np.int32))
        except Exception as e:
            # Se un campo dà errore (come le flag bit-to-bit), lo saltiamo per non bloccare il salvataggio
            print(f"[WARNING] Impossibile scrivere la feature {col_name}: {e}")

    # 4. Classificazione Predetta
    las.classification = Y.astype(np.uint8)
    
    las.write(filename)
    print(f"---> Salvataggio completato: {filename}")

def save_model(model, filename):
    ''' Save the trained machine learning model

        Attribures:
            model       :   model to save
            filename    :   name of the file where the model is saved
    '''
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def get_voxel_size_from_las(filepath, budget_bytes, sample_size=2000000):
    """
    Median distance to the nearest neighbour (suggested voxel size). Exact when the points and their KD-tree fit the
    budget (~80 B/point); otherwise estimated on a random sample of `sample_size` points, rescaled by the density
    ratio (the distance scales with (n_sample / n)^(1/3) in a 3D cloud), and logged as an estimate.
    """
    las = pcm.LasFile(filepath)
    rows = las.block_rows_for(budget_bytes * 0.1, per_row_extra=24)
    exact = las.n * 80 <= budget_bytes * 0.5
    rng = np.random.RandomState(1)
    keep = None if exact else min(1.0, sample_size / float(las.n))
    parts = []
    for _, blk in las.blocks(['X', 'Y', 'Z'], rows):
        sel = slice(None) if keep is None else rng.random_sample(len(blk)) < keep
        c = np.column_stack([np.asarray(blk[k][sel], dtype=np.float64) * las.scale[i] for i, k in enumerate('XYZ')])
        parts.append(c)
    coords = np.concatenate(parts)
    tree = KDTree(coords)
    distanze, _ = tree.query(coords, k=2)
    d = float(np.median(distanze[:, 1]))
    if keep is not None:
        d *= (len(coords) / float(las.n)) ** (1.0 / 3.0)
        print(f"[Info] voxel distance estimated on a sample of {len(coords)} of {las.n} points (memory budget)", flush=True)
    return d


def main():
    parser = argparse.ArgumentParser(description='Train the random forest model.')
    parser.add_argument('--selected_features', nargs="+", required = True, help='Selected feature for training')
    parser.add_argument('--training_filepath', required = True, help='Path to the training file (.las) [f1, ..., fn, c]')
    parser.add_argument('--val_filepath', required = True, help='Path to the test file (.las) [f1, ..., fn, c]')
    parser.add_argument('--n_jobs', required = True, help='Number of threads used to train the model', type=int)
    parser.add_argument('--n_estimators', required = True, help='Number of trees (e.g. 200)', type=int)
    parser.add_argument('--max_depth', help='Maximum depth of each tree', type=int)
    parser.add_argument('--min_samples_split', help='Minimum samples required to split a node', type=int)
    parser.add_argument('--max_features', help='Number of features per split', type=str)
    parser.add_argument('--use_gpu', help='Try to use GPU-accelerated training (cuML)', action='store_true')
    parser.add_argument('--output_training_name', required = True, help='Name of the predicted test file')
    parser.add_argument('--model_savepath', help='Path to save the model')
    parser.add_argument('--report_savepath', help='Path to save the training report')
    parser.add_argument('--max_training_points', type=int, default=None,
                        help='Points kept from the training and validation sets (stratified sampling by class when they have more); default: what the memory budget allows')
    pcm.add_common_args(parser)
    args= parser.parse_args()

    # Crea cartelle per output, modello e report
    for path in [args.output_training_name, args.model_savepath, args.report_savepath]:
        if path:
            output_dir = os.path.dirname(path)
            if output_dir and not os.path.exists(output_dir):
                print(f"Create output folder: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

    # PARAMETERS 
    selected_features = args.selected_features
    training_filepath = args.training_filepath
    val_filepath = args.val_filepath
    n_jobs = args.n_jobs
    n_estimators = args.n_estimators if args.n_estimators else 200
    max_depths = args.max_depth if args.max_depth else 15
    min_samples_split=args.min_samples_split if args.min_samples_split else 20
    max_features=args.max_features if args.max_features else "sqrt"
    use_gpu = args.use_gpu
    output_training_name = args.output_training_name
    model_savepath = args.model_savepath
    report_savepath = args.report_savepath if args.report_savepath else os.path.join(os.path.dirname(output_training_name), 'report_RF.txt')
    class_las_index = "labels"

    feat_to_use = []
    
    total_start = time.time()
    t0 = time.time()

    budget = pcm.memory_budget_bytes(args.memory_budget)
    print(f"Memory budget: {budget // 1048576} MB")
    # Only the selected features are loaded (and the two sets share the budget)
    print("\nLoading training data...")
    X_train, Y_train, header, _ = read_las_data(training_filepath, class_las_index, selected_features, budget // 2,
                                                args.max_training_points, what='training set')

    # Calculate voxel distance
    suggested_voxel = get_voxel_size_from_las(training_filepath, budget // 2)

    print("\nLoading validation data...")
    X_test, Y_test, header_val, meta = read_las_data(val_filepath, class_las_index, selected_features, budget // 2,
                                                     args.max_training_points, what='validation set')
    if header_val != header:
        raise SystemExit(f"The training and validation sets do not have the same features: {header} vs {header_val}")
    feat_to_use = list(range(len(header)))      # the matrices already hold only the selected features

    t1 = time.time()
    tot_sec = round(t1 - t0, 2)
    load_min = int(tot_sec // 60)
    load_sec = int(tot_sec % 60)
    if load_min > 0:
        print(f'\n---> Loading time {load_min} min {load_sec} sec')
    else:
        print(f'\n---> Loading time {load_sec} sec')
    print('\nStatistics:\n- Training samples: {}\n- Testing samples: {}\n- Features selected: {}'.format(len(Y_train), len(Y_test), selected_features))

    # Use the header for feature importance visualization
    if header:
        headers = [header[i] for i in feat_to_use]  # Select only the headers corresponding to feat_to_use
    else:
        headers = ['col_{}'.format(i) for i in feat_to_use]  # Fallback: use column indices from feat_to_use

    ''' ***************************************** TRAINING ************************************** '''
    # Fixed RF200 configuration: 200 trees, max_depth=15, min_samples_split=20, max_features='sqrt'
    print(f'\nTraining the model n_jobs={n_jobs}, n_estimators={n_estimators}, max_depth={max_depths}, min_samples_split={min_samples_split}, max_features={max_features} ...')  
    tot_train_sec = 0
    tot_metrics_sec = 0
    
    t2 = time.time()
    
    # X_train[:, feat_to_use] specify only feature needed
    model, feats_raw = train_model(X_train, Y_train, n_jobs=n_jobs, use_gpu=use_gpu,
                        n_estimators=n_estimators,
                        max_depth=max_depths,
                        min_samples_split=min_samples_split,
                        max_features=max_features
                        )  # Uses RF200 defaults

    t3 = time.time()
    tot_train_sec = round(t3 - t2, 2)
    train_min = int(tot_train_sec // 60)
    train_sec = int(tot_train_sec % 60)
    if train_min > 0:
        print(f'---> Training time {train_min} min {train_sec} sec')
    else:
        print(f'---> Training time {train_sec} sec')
    
    print('\nEvaluating on validation set...')
    # Predict depending on whether we used cuML (GPU) or scikit-learn (CPU)
    if use_gpu and GPU_AVAILABLE and cuRF is not None:
        Y_test_pred = cp.asnumpy(model.predict(cp.asarray(X_test)))
    else:
        Y_test_pred = model.predict(X_test)             # Test the model, using only the specified features
    # print(f'\nSaving {output_training_name}')
    # write_classification_las(X_test, Y_test_pred, output_training_name, header, meta)

    # Compute metrics
    print('\nComputing metrics...')
    Precision_pc = precision_score(Y_test, Y_test_pred, average=None)
    Precision = precision_score(Y_test, Y_test_pred, average='weighted')
    Recall_pc = recall_score(Y_test, Y_test_pred, average=None)
    Recall = recall_score(Y_test, Y_test_pred, average='weighted')
    f1_pc = f1_score(Y_test, Y_test_pred, average=None)
    f1 = f1_score(Y_test, Y_test_pred, average='weighted')
    IoU_pc = jaccard_score(Y_test, Y_test_pred, average=None)
    IoU = jaccard_score(Y_test, Y_test_pred, average='weighted')
    con_mat = confusion_matrix(Y_test, Y_test_pred)
    acc = accuracy_score(Y_test, Y_test_pred)
    t4 = time.time()

    tot_metrics_sec = round(t4 - t3, 2)
    metr_min = int(tot_metrics_sec // 60)
    metr_sec = int(tot_metrics_sec % 60)
    if metr_min > 0:
        print(f'---> Computing Metrics time: {metr_min} min {metr_sec} sec')
    else:
        print(f'---> Computing Metrics time {metr_sec} sec')
    
    # Sort features by importance in descending order
    # Try to get feature importances; cuML may expose a similar attribute
    if feats_raw is not None and len(feats_raw) == len(headers):
        feats = feats_raw
    else:
        print(f"[WARNING] feats shape={None if feats_raw is None else len(feats_raw)}, headers={len(headers)}")
        feats = np.zeros(len(headers))
    sorted_features = sorted(zip(headers, feats), key=lambda x: x[1], reverse=True)
    
    # Print results
    print('\n- Confusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('\n- Feature importance:')
    for name, importance in sorted_features:
        print(f'{name}: {importance:.4f}')

    #Print report with results
    report_fname = report_savepath

    print(f'\nCheck the complete report in the folder: {report_fname}')
    ''' ******************************************************************************************** '''

    tot_sec = round(t4 - total_start, 2)
    tot_min = int(tot_sec // 60)
    tot_sec = int(tot_sec % 60)

    np.set_printoptions(precision=4)
    file = open(report_fname,"w") 
    file.write("Overall Accuracy\n\n")
    file.write('{:.4f}'.format(acc))
    file.write("\n\nPrecision per class\n\n")
    file.write(str(Precision_pc))   
    file.write("\n\nAverage Precision\n\n")
    file.write('{:.4f}'.format(Precision_pc.mean()))
    file.write("\n\nWeighted Precision\n\n")
    if np.ndim(Precision) == 0:  # in case is a single value
         file.write('{:.4f}'.format(Precision))
    else:  # otherwise can be computed the mean
        file.write('{:.4f}'.format(Precision.mean()))
    file.write("\n\nRecall per class\n\n")
    file.write(str(Recall_pc))
    file.write("\n\nAverage Recall\n\n")
    file.write('{:.4f}'.format(Recall_pc.mean()))
    file.write("\n\nWeighted Recall\n\n")
    if np.ndim(Recall) == 0:  # in case is a single value
         file.write('{:.4f}'.format(Recall))
    else:  # otherwise can be computed the mean
        file.write('{:.4f}'.format(Recall.mean()))
    file.write("\n\nF1 Scores per class\n\n") 
    file.write(str(f1_pc))
    file.write("\n\nAverage F1 Score\n\n") 
    file.write('{:.4f}'.format(f1_pc.mean()))
    file.write("\n\nWeighted F1 Score\n\n") 
    if np.ndim(f1) == 0:  # in case is a single value
         file.write('{:.4f}'.format(f1))
    else:  # otherwise can be computed the mean
        file.write('{:.4f}'.format(f1.mean()))
    file.write("\n\nIoU per class\n\n")
    file.write(str(IoU_pc))
    file.write("\n\nAverage IoU\n\n")
    file.write('{:.4f}'.format(IoU_pc.mean()))
    file.write("\n\nWeighted IoU\n\n")
    if np.ndim(IoU) == 0:  # in case is a single value
         file.write('{:.4f}'.format(IoU))
    else:  # otherwise can be computed the mean
        file.write('{:.4f}'.format(IoU.mean()))
    file.write("\n\nConfusion Matrix\n\n")
    file.write(str(con_mat))
    file.write("\n\nFeature importance\n\n")
    for name, importance in sorted_features:
        file.write(f'{name}: {importance:.4f}\n')
    file.write(f"\n\nVoxel distance value = {suggested_voxel:.4f} \n")
    file.write("\n\nTiming\n\n")

    if load_min > 0:
        file.write(f'Loading time:              {load_min} min {load_sec} sec\n')
    else:
        file.write(f'Loading time:              {load_sec} sec\n')
    
    if train_min > 0:
        file.write(f'Training time:             {train_min} min {train_sec} sec\n')
    else:
        file.write(f'Training time:             {train_sec} sec\n')

    if metr_min > 0:
        file.write(f'Prediction + Metrics time: {metr_min} min {metr_sec} sec\n')
    else:
        file.write(f'Prediction + Metrics time: {metr_sec} sec\n')

    if tot_min > 0:
        file.write(f'Total time:                {tot_min} min {tot_sec} sec\n')
    else:
        file.write(f'Total time:                {tot_sec} sec\n')

    file.close()

    print(f"Model saved on {model_savepath}")
    save_model(model, model_savepath )
    
    if tot_min > 0:
        print(f'---> Total time {tot_min} min {tot_sec} sec')
    else:
        print(f'---> Total time {tot_sec} sec')

if __name__== '__main__':
    main()