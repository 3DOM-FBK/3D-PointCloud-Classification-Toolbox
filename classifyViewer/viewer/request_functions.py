from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, FileResponse, Http404,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .functions import launch_training_RF, launch_classify_RF, subsampling_point_cloud, stop_processes, get_voxel_size, check_point_id, inspect_las_header
from .functions import mesh_to_point_cloud, ply_to_las, feature_extraction, build_pointcloud, update_pointcloud_columns, PointCloudMismatch, split_las_by_store, extract_segment_las
import base64
import os
import json
import traceback
import re
import datetime
import zipfile
import tempfile
import shutil
import zlib
import numpy as np
from django.conf import settings
from io import BytesIO


def _get_working_dir():
    working = settings.RUNTIME_DATA_ROOT / 'working'
    working.mkdir(parents=True, exist_ok=True)
    return str(working)


def _get_working_file(*parts):
    return os.path.join(_get_working_dir(), *parts)


def _get_models_dir():
    models = settings.RUNTIME_DATA_ROOT / 'models'
    models.mkdir(parents=True, exist_ok=True)
    return str(models)


def _runtime_relative_path(*parts):
    """Return path relative to BASE_DIR, e.g. runtime_data/working/features.las"""
    base = settings.RUNTIME_DATA_ROOT.relative_to(settings.BASE_DIR).as_posix()
    return '/'.join([base, *parts])


# Single chunked point cloud folder under runtime_data/working (keep in sync with PC_DIR in functions.js)
PC_DIR = 'pc'


def _las_point_count(las_path):
    """Number of points from the LAS header (legacy count, or the 64-bit one of LAS 1.4)."""
    import struct
    with open(las_path, 'rb') as f:
        head = f.read(375)
    count, = struct.unpack_from('<I', head, 107)
    if count == 0 and head[25] >= 4 and len(head) >= 255:
        count, = struct.unpack_from('<Q', head, 247)
    return count


def _annotations_relative_path():
    """Default annotations store: runtime_data/working/annotations.bin"""
    return _runtime_relative_path('working', 'annotations.bin')


def _resolve_annotations_path(annotations_path=None):
    """
    Resolve the annotations.bin path (relative to BASE_DIR) to a validated absolute path.
    Raises ValueError if it escapes BASE_DIR, FileNotFoundError if the file is missing.
    Returns (relative_path, absolute_path).
    """
    rel = annotations_path or _annotations_relative_path()
    base = os.path.normpath(str(settings.BASE_DIR))
    abs_path = os.path.normpath(os.path.join(base, rel))
    if not abs_path.startswith(base):
        raise ValueError("Invalid annotations_path")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            "annotations.bin not found: no annotations have been saved yet. "
            "Create segments/classes in the viewer first."
        )
    return rel, abs_path

@csrf_exempt
def launch_RF_training(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] Launch RF training: ", request.body[:200])
            data = json.loads(request.body)
            launch_training_RF(data)
            print("\n")
            return JsonResponse({"status": 'success', "message": "RF training launched successfully."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Launch RF training ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def launch_RF_classify(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] Launch RF classify", request.body[:200]) 
            data = json.loads(request.body)
            launch_classify_RF(data)
            print("\n")
            return JsonResponse({"status": 'success', "message": "RF classify launched successfully."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Launch RF classify ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def subsample_pc(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] Subsample Point Cloud:", request.body[:200]) 
            data = json.loads(request.body)

            file_path = data['file_path']
            out_path = data['out_path']
            voxel_size = data['voxel_size'] 

            output_file_path = subsampling_point_cloud(file_path, out_path, voxel_size)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Subsampling completed.", "output_file_path": output_file_path})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Subsample Point Cloud ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405) 

@csrf_exempt
def get_model_voxel_size(request):
    """Retrieve the voxel distance value from a model's report file."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_dir = data.get('model_dir', '')
            if not model_dir:
                return JsonResponse({"status": 'error', "message": "Missing 'model_dir'."}, status=400)
            
            voxel_size = get_voxel_size(model_dir)
            
            return JsonResponse({
                "status": 'success',
                "voxel_size": voxel_size
            })

        except Exception as e:
            print("\n[REQUEST FUNCTION] get_model_voxel_size ERROR " + str(e))
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def mesh2pc(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] Mesh to Point Cloud:", request.body[:200]) 
            data = json.loads(request.body)

            file_path = data['file_path']
            out_path = data['out_path']
            num_points = data['num_points']
            # sampling_method = data['sampling_method']

            mesh_to_point_cloud(file_path, out_path, num_points=num_points)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Mesh to Point Cloud completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Mesh to Point Cloud ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405) 

@csrf_exempt
def checking_point_id(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] CHECK POINT ID:", request.body[:200]) 
            data = json.loads(request.body)

            input_path = data['input_path']
            output_path = data['output_path']

            check_point_id(input_path, out_path=output_path)
            print("\n")

            return JsonResponse({"status": 'success', "message": "CHECK POINT ID completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] CHECK POINT ID ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405) 


@csrf_exempt
def inspect_las_input(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] INSPECT LAS INPUT:", request.body[:200])
            data = json.loads(request.body)

            file_path = data['file_path']
            info = inspect_las_header(file_path)

            return JsonResponse({"status": 'success', **info})

        except Exception as e:
            print("\n[REQUEST FUNCTION] INSPECT LAS INPUT ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def ply2las(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] PLY to LAS:", request.body[:200]) 
            data = json.loads(request.body)

            file_path = data['file_path']
            out_path = data['out_path']

            ply_to_las(file_path, out_path=out_path)
            print("\n")

            return JsonResponse({"status": 'success', "message": "PLY to LAS completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] PLY to LAS ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405) 

@csrf_exempt
def feat_extraction(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] FEATURE EXTRACTION:", request.body[:200]) 
            data = json.loads(request.body)

            input_filepath = data['input_filepath']
            output_filepath = data['output_filepath']
            feature_list = data['feature_list']
            radius_list = data['radius_list']
            sampling = data.get('sampling', 0)
            use_gpu = data.get('use_gpu', True)
        
            feature_extraction(input_filepath, output_filepath, feature_list, radius_list, sampling, use_gpu=use_gpu)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Feature extraction completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Feature extraction ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def build_pointcloud_view(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] BUILD POINT CLOUD:", request.body[:200])
            data = json.loads(request.body)

            input_filepath = data['input_filepath']
            output_filepath = data['output_filepath']

            version = build_pointcloud(input_filepath, output_filepath)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Point cloud built.", "version": version})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Build point cloud ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def update_pointcloud_columns_view(request):
    """
    Rewrites attribute columns of the current point cloud (the geometry is untouched).

    JSON body (paths relative to BASE_DIR, like the other endpoints):
      pc_dir               : point cloud folder (default runtime_data/working/pc)
      las_filepath + only  : write the Extra Bytes `only` of the LAS (or all of them when `only` is empty)
      prediction_filepath  : write the 'prediction' column from a classified LAS
      drop_all             : remove every column first
      prune                : with las_filepath, drop the columns that are not in the LAS any more

    Answers 409 {"status": "mismatch"} when the classified LAS does not belong to the point cloud.
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] UPDATE POINT CLOUD COLUMNS:", request.body[:200])
            data = json.loads(request.body)
            pc_dir = data.get('pc_dir') or _runtime_relative_path('working', PC_DIR)
            las_filepath = data.get('las_filepath')
            prediction = data.get('prediction_filepath')
            only = data.get('only') or None
            drop_all = bool(data.get('drop_all'))
            prune = bool(data.get('prune'))
            if not (las_filepath or prediction or drop_all):
                return JsonResponse({'status': 'error', 'message': 'Nothing to update'}, status=400)

            update_pointcloud_columns(pc_dir, las_filepath=las_filepath, only=only,
                                      prediction_filepath=prediction, drop_all=drop_all, prune=prune)
            print("\n")
            return JsonResponse({"status": 'success', "message": "Point cloud columns updated."})

        except PointCloudMismatch as e:
            print("\n[REQUEST FUNCTION] Update point cloud columns MISMATCH " + str(e))
            return JsonResponse({'status': 'mismatch', 'message': str(e)}, status=409)
        except Exception as e:
            print("\n[REQUEST FUNCTION] Update point cloud columns ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def stop_process(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] STOP PROCESS:") 

            stop_processes()

            return JsonResponse({"status": 'success', "message": "Process stopped successfully."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Stop process ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def save_file(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] Save file: ", request.body[:200]) 
            
            data = json.loads(request.body)
            filepath = data['filepath']
            
            # print("📂 Filepath:", filepath)
            # print("📊 Data size:", len(data['data']), "bytes (base64)")
            
            file_data = base64.b64decode(data['data'])
            
            # 🔧 Control path is absolute
            if not os.path.isabs(filepath):
                # If filepath is relative, use it in the project folder
                filepath = os.path.join(settings.BASE_DIR, filepath)
            
            # print("[Save file] Full path:", filepath)
            
            # Create folder if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                f.write(file_data)
            
            print(f"[REQUEST FUNCTION] File saved: {filepath} ({len(file_data)} bytes)")
            
            return JsonResponse({'status': 'success', 'filepath': filepath})
            
        except Exception as e:
            # Print the full error for debugging.
            print("\n[REQUEST FUNCTION] Save file ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def _split_las_by_binary(request):
    """
    Split LAS point cloud by segment annotations from the annotations.bin store.

    POST body (JSON):
        las_path         - path to features.las
        annotations_path - path to annotations.bin (default: working/annotations.bin)
        output_dir       - destination directory for output segment_*.las files
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] SPLIT LAS BY ANNOTATIONS:", request.body[:200])
            data = json.loads(request.body)

            las_path   = data['las_path']
            output_dir = data['output_dir']
            exclude_unclassified = bool(data.get('exclude_unclassified', False))
            annotations_path, _ = _resolve_annotations_path(data.get('annotations_path'))

            split_las_by_store(las_path, annotations_path, output_dir, exclude_unclassified=exclude_unclassified)

            # Rename segment files to training.las / validation.las if mapping provided
            segment_names = data.get('segment_names')  # e.g. {"1": "training", "2": "validation"}
            if segment_names:
                abs_outdir = os.path.abspath(os.path.join(settings.BASE_DIR, output_dir) if not os.path.isabs(output_dir) else output_dir)
                for seg_id_str, role_name in segment_names.items():
                    src = os.path.join(abs_outdir, f"segment_{seg_id_str}.las")
                    dst = os.path.join(abs_outdir, f"{role_name}.las")
                    if os.path.isfile(src):
                        os.replace(src, dst)
                        print(f"[SPLIT] Renamed {src} -> {dst}")
                    else:
                        print(f"[SPLIT] Warning: expected {src} not found")
            print("\n")

            return JsonResponse({"status": 'success', "message": "Split LAS completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Split LAS ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

def models_list(request):
    """Return a list of all trained models found in runtime_data/models/."""
    if request.method == 'GET':
        try:
            models_root = _get_models_dir()
            result = []

            if os.path.isdir(models_root):
                for name in sorted(os.listdir(models_root)):
                    model_dir = os.path.join(models_root, name)
                    pkl_path  = os.path.join(model_dir, 'model.pkl')
                    if not os.path.isdir(model_dir) or not os.path.isfile(pkl_path):
                        continue
                    stat = os.stat(pkl_path)
                    created = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                    size_mb = round(stat.st_size / (1024 * 1024), 2)
                    result.append({
                        'name': name,
                        'path': _runtime_relative_path('models', name, 'model.pkl'),
                        'created': created,
                        'size_mb': size_mb,
                    })

            return JsonResponse({'status': 'success', 'models': result})

        except Exception as e:
            print("\n[REQUEST FUNCTION] models_list ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


def model_exists(request):
    """Check whether a model folder already exists under runtime_data/models/{name}/."""
    if request.method == 'GET':
        try:
            name = request.GET.get('name', '').strip()
            if not name:
                return JsonResponse({'exists': False})

            model_dir = os.path.join(_get_models_dir(), name)
            exists = os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, 'model.pkl'))

            return JsonResponse({'exists': exists})

        except Exception as e:
            print("\n[REQUEST FUNCTION] model_exists ERROR " + str(e))
            return JsonResponse({'exists': False})

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def delete_model(request):
    """Delete a trained model folder from runtime_data/models/{name}/."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name', '').strip()

            if not name:
                return JsonResponse({'status': 'error', 'message': 'No model name provided'}, status=400)

            # Prevent path traversal
            if '/' in name or '\\' in name or '..' in name:
                return JsonResponse({'status': 'error', 'message': 'Invalid model name'}, status=400)

            model_dir = os.path.join(_get_models_dir(), name)

            if not os.path.isdir(model_dir):
                return JsonResponse({'status': 'error', 'message': 'Model not found'}, status=404)

            shutil.rmtree(model_dir)
            print(f"\n[REQUEST FUNCTION] Model '{name}' deleted: {model_dir}")

            return JsonResponse({'status': 'success', 'message': f"Model '{name}' deleted successfully."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] delete_model ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def extract_segment_las_view(request):
    """
    Extract all points for a single segment from features.las into a new .las
    file, using the annotations.bin store for annotation lookup.

    POST body (JSON):
        las_path         - path to features.las
        annotations_path - path to annotations.bin (default: working/annotations.bin)
        seg_id           - integer segment ID to extract
        out_path         - destination .las path
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] EXTRACT SEGMENT LAS:", request.body[:200])
            data = json.loads(request.body)

            las_path   = data['las_path']
            seg_id     = int(data['seg_id'])
            out_path   = data['out_path']
            annotations_path, _ = _resolve_annotations_path(data.get('annotations_path'))

            extract_segment_las(las_path, annotations_path, seg_id, out_path)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Segment extraction completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] EXTRACT SEGMENT LAS ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


def read_text_file(request):
    """Read a text file from the server and return its content."""
    if request.method == 'GET':
        try:
            file_path = request.GET.get('path', '')
            if not file_path:
                return JsonResponse({'status': 'error', 'message': 'No path provided'}, status=400)

            # Make absolute path relative to project root
            if not os.path.isabs(file_path):
                file_path = os.path.join(settings.BASE_DIR, file_path)

            if not os.path.exists(file_path):
                return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)

            # If it's a directory, find the first .txt file inside it
            if os.path.isdir(file_path):
                txt_files = [f for f in os.listdir(file_path) if f.lower().endswith(".txt")]
                if txt_files:
                    file_path = os.path.join(file_path, txt_files[0])
                else:
                    return JsonResponse({'status': 'error', 'message': 'No .txt file found in directory'}, status=404)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return JsonResponse({'status': 'success', 'content': content})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Read text file ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def serve_range_file(request, filepath):
    """
    Serve the chunked point cloud files (geom.bin, col/*.bin, meta.json) with HTTP Range support.

    Django does not handle Range by itself (ticket #22479). The file is opened and positioned at the
    first requested byte, then handed to FileResponse: the WSGI server receives it through
    wsgi.file_wrapper and, with an explicit Content-Length, Gunicorn sendfile()s exactly that slice
    (zero-copy, no Python loop). A Python generator tops out at ~65 MB/s because of the GIL.

    Each request opens its own descriptor on purpose: sendfile uses the descriptor's current offset,
    so sharing one between concurrent requests would be racy (and open() costs microseconds).

    With ?v=<token> in the query the response is cacheable forever (the token changes whenever the
    file content changes); otherwise the browser has to revalidate.
    """
    ALLOWED_EXTENSIONS = ('.bin', '.json')
    BASE_DATA_DIR = os.path.normpath(str(settings.RUNTIME_DATA_ROOT))

    # We lstrip('/') to ensure os.path.join doesn't treat it as an absolute path
    full_path = os.path.normpath(os.path.join(BASE_DATA_DIR, filepath.lstrip('/')))

    # Prevent directory traversal
    if os.path.commonpath([full_path, BASE_DATA_DIR]) != BASE_DATA_DIR:
        raise Http404("Access denied")

    if not os.path.isfile(full_path):
        raise Http404(f"File not found: {filepath}")

    ext = os.path.splitext(full_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise Http404("File type not allowed")

    file_size = os.path.getsize(full_path)
    content_type = 'application/json' if ext == '.json' else 'application/octet-stream'

    status = 200
    start, length = 0, file_size
    range_header = request.META.get('HTTP_RANGE', '')
    range_match = re.match(r'bytes=(\d*)-(\d*)\s*$', range_header) if range_header else None
    if range_match and (range_match.group(1) or range_match.group(2)):
        if range_match.group(1):
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
        else:  # suffix range: last N bytes
            start = max(0, file_size - int(range_match.group(2)))
            end = file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            response = HttpResponse(status=416)  # Range Not Satisfiable
            response['Content-Range'] = f'bytes */{file_size}'
            return response
        length = end - start + 1
        status = 206

    f = open(full_path, 'rb')
    f.seek(start)
    response = FileResponse(f, status=status, content_type=content_type)
    # FileResponse derives Content-Length from the file size minus the current position: override it
    # so that only the requested slice is sent (also read by Gunicorn's sendfile as the byte count).
    response['Content-Length'] = length
    if status == 206:
        response['Content-Range'] = f'bytes {start}-{start + length - 1}/{file_size}'
    response['Accept-Ranges'] = 'bytes'
    response['Access-Control-Allow-Origin'] = '*'
    if request.GET.get('v'):
        response['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        response['Cache-Control'] = 'no-cache'
    return response


def serve_runtime_file(request, filepath):
    """Serve runtime data files (LAS, JSON, ...) without Range support (streamed)."""
    ALLOWED_EXTENSIONS = ('.las', '.bin', '.json', '.txt')
    BASE_DATA_DIR = str(settings.RUNTIME_DATA_ROOT)

    full_path = os.path.normpath(os.path.join(BASE_DATA_DIR, filepath.lstrip('/')))

    if not full_path.startswith(os.path.normpath(BASE_DATA_DIR)):
        raise Http404("Access denied")

    if not os.path.isfile(full_path):
        raise Http404(f"File not found: {filepath}")

    _, ext = os.path.splitext(full_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise Http404("File type not allowed")

    content_type = 'application/octet-stream'
    if ext.lower() == '.json':
        content_type = 'application/json'
    elif ext.lower() == '.txt':
        content_type = 'text/plain'

    # FileResponse streams the file in chunks (no full read in memory)
    response = FileResponse(open(full_path, 'rb'), content_type=content_type)
    response['Access-Control-Allow-Origin'] = '*'
    return response


@csrf_exempt
def clear_data(request):
    """
    Clears all files and subdirectories in static/viewer/data/working/
    leaving static/viewer/data/models/ untouched.
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] CLEAR DATA") 
            working_dir = _get_working_dir()
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir, exist_ok=True)

            return JsonResponse({"message": "Working directory cleared successfully"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)


@csrf_exempt
def upload_data(request):
    """
    Endpoint for uploading a point cloud file (.ply, .las, .laz, .glb) 
    to the static/viewer/data directory.
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] UPLOAD DATA")
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({"error": "No file provided"}, status=400)

            # Define working data directory
            data_dir = _get_working_dir()
            os.makedirs(data_dir, exist_ok=True)

            file_path = os.path.join(data_dir, uploaded_file.name)
            
            # Save file
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # # --- Validation & Enhancement for LAS files ---
            # if uploaded_file.name.lower().endswith('.las'):
            #     try:
            #         print(f"Validating upload: {file_path}")
            #         # check_point_id(in, out=None) will overwrite by default with my change
            #         # or I can pass a separate path for safety.
            #         fixed_path = file_path.replace('.las', '_fix.las')
            #         result_path = check_point_id(file_path, out_path=fixed_path)
                    
            #         if result_path == fixed_path:
            #             # File was actually changed/fixed, replace the original
            #             os.replace(fixed_path, file_path)
            #             print(f"File validated and enhanced: {file_path}")
            #         else:
            #             # No change needed, cleanup temp if it was created
            #             if os.path.exists(fixed_path):
            #                 os.remove(fixed_path)
            #             print("File already valid.")
            #     except Exception as ex:
            #         print(f"Validation error (ignored): {ex}")

            #     features_las_path = os.path.join(data_dir, 'features.las')
            #     if os.path.abspath(file_path) != os.path.abspath(features_las_path):
            #         shutil.copy2(file_path, features_las_path)
            #     print(f"Canonical features.las prepared: {features_las_path}")

            return JsonResponse({
                "message": "File uploaded successfully",
                "filename": uploaded_file.name,
                "rel_path": uploaded_file.name
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)


@csrf_exempt
def backup_pointcloud(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] BACKUP POINT CLOUD")
            features_las = _get_working_file('features.las')
            backup_las = _get_working_file('pointcloud_backup.las')

            if not os.path.isfile(features_las):
                return JsonResponse({"error": "features.las not found"}, status=404)

            shutil.copy2(features_las, backup_las)
            return JsonResponse({
                "status": "success",
                "message": "Point cloud backup created",
                "backup_path": _runtime_relative_path('working', 'pointcloud_backup.las')
            }, status=200)
        except Exception as e:
            print("\n[REQUEST FUNCTION] BACKUP POINT CLOUD ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def restore_pointcloud_backup(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] RESTORE POINT CLOUD BACKUP")

            working_dir = _get_working_dir()
            features_las = _get_working_file('features.las')
            backup_las = _get_working_file('pointcloud_backup.las')

            if not os.path.isfile(backup_las):
                return JsonResponse({"error": "point cloud backup not found"}, status=404)

            os.makedirs(working_dir, exist_ok=True)
            shutil.copy2(backup_las, features_las)

            # The cloud goes back to the backup state: stored annotations no longer apply
            stale_annotations = _get_working_file('annotations.bin')
            if os.path.isfile(stale_annotations):
                os.remove(stale_annotations)

            # Legacy unified store from older versions (no longer used)
            legacy_pcbin = _get_working_file('features.pcbin')
            if os.path.isfile(legacy_pcbin):
                os.remove(legacy_pcbin)

            # Geometry and POINT_ID are identical to the backup state: only the columns go back.
            # A different point count means a different cloud: rebuild the geometry.
            pc_rel = _runtime_relative_path('working', PC_DIR)
            las_rel = _runtime_relative_path('working', 'features.las')
            meta_path = _get_working_file(PC_DIR, 'meta.json')
            rebuilt = True
            version = None
            if os.path.isfile(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                if int(meta.get('points', -1)) == _las_point_count(backup_las):
                    update_pointcloud_columns(pc_rel, las_filepath=las_rel, drop_all=True)
                    version = str(meta['version'])
                    rebuilt = False
            if rebuilt:
                version = build_pointcloud(las_rel, pc_rel)

            return JsonResponse({
                "status": "success",
                "message": "Point cloud restored from backup",
                "las_path": las_rel,
                "version": version,
                "rebuilt": rebuilt
            }, status=200)
        except Exception as e:
            print("\n[REQUEST FUNCTION] RESTORE POINT CLOUD BACKUP ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def export_mapping(request):
    """
    Save the user annotations (segments/classes) into working/annotations.bin.

    The client buffer only contains the segments requested by the caller (not
    necessarily every existing segment), so the store is PATCHED: points with
    seg != 0 in the buffer overwrite the stored state, all others are left untouched.
    If no store exists yet (or its size differs, i.e. the cloud changed), the buffer
    becomes the store. The upload is streamed to a temp file in chunks (gunzipped on
    the fly when encoding=gzip), patched with numpy on a memmap (no per-point Python
    loop) and moved atomically.

    Expects FormData with:
    - 'buffer'      : Binary blob (2 bytes per point: segment_id+1, class_id)
    - 'point_count' : Integer (total number of points in the buffer)
    - 'encoding'    : optional, 'gzip' if the buffer is gzip-compressed

    Returns JSON:
    - annotations_path : relative path of annotations.bin
    - point_count      : number of points covered by the buffer
    """
    if request.method == 'POST':
        tmp_path = None
        try:
            print("\n[REQUEST FUNCTION] SAVE ANNOTATIONS (annotations.bin)")
            buffer_file = request.FILES.get('buffer')
            point_count_str = request.POST.get('point_count')
            encoding = (request.POST.get('encoding') or '').lower()

            if not buffer_file:
                return JsonResponse({"error": "Missing 'buffer' binary data"}, status=400)
            if not point_count_str:
                return JsonResponse({"error": "Missing 'point_count'"}, status=400)

            try:
                point_count = int(point_count_str)
            except ValueError:
                return JsonResponse({"error": "Invalid point_count (must be integer)"}, status=400)
            if point_count <= 0:
                return JsonResponse({"error": "Invalid point_count (must be > 0)"}, status=400)
            if encoding not in ('', 'gzip'):
                return JsonResponse({"error": f"Unsupported encoding '{encoding}'"}, status=400)

            expected_size = point_count * 2
            rel_path = _annotations_relative_path()
            abs_path = _get_working_file('annotations.bin')
            tmp_path = abs_path + '.tmp'

            # Stream to the temp file, never holding the whole buffer in memory
            written = 0
            with open(tmp_path, 'wb') as out:
                if encoding == 'gzip':
                    decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    for chunk in buffer_file.chunks():
                        data = decomp.decompress(chunk)
                        written += len(data)
                        if written > expected_size:
                            break
                        out.write(data)
                    else:
                        data = decomp.flush()
                        written += len(data)
                        out.write(data)
                else:
                    for chunk in buffer_file.chunks():
                        written += len(chunk)
                        if written > expected_size:
                            break
                        out.write(chunk)

            if written != expected_size:
                os.remove(tmp_path)
                tmp_path = None
                return JsonResponse({
                    "error": f"Buffer size mismatch: expected {expected_size} bytes, got {'more than ' if written > expected_size else ''}{written}"
                }, status=400)

            # Patch semantics: the client only exports the segments it was asked for
            # (e.g. training/validation), not necessarily every existing segment. So
            # only points with seg != 0 in the new buffer overwrite the stored state.
            # Vectorized with numpy on a memmap, in chunks (never a per-point Python loop).
            patched_path = None
            if os.path.isfile(abs_path) and os.path.getsize(abs_path) == expected_size:
                patched_path = abs_path + '.patch.tmp'
                shutil.copyfile(abs_path, patched_path)
                old = np.memmap(patched_path, dtype=np.uint8, mode='r+', shape=(point_count, 2))
                new = np.memmap(tmp_path, dtype=np.uint8, mode='r', shape=(point_count, 2))
                chunk = 8_000_000
                for start in range(0, point_count, chunk):
                    stop = min(start + chunk, point_count)
                    new_c = new[start:stop]
                    mask = new_c[:, 0] != 0
                    if mask.any():
                        old_c = old[start:stop]
                        old_c[mask] = new_c[mask]
                old.flush()
                del old, new
                os.replace(patched_path, abs_path)
                os.remove(tmp_path)
            else:
                # No previous store (or the cloud changed size): the new buffer becomes the store
                os.replace(tmp_path, abs_path)
            tmp_path = None

            print(f"[REQUEST FUNCTION] annotations.bin saved: {point_count} points → {abs_path}")

            return JsonResponse({
                "annotations_path": rel_path,
                "point_count": point_count,
            }, status=200)

        except Exception as e:
            print(f"[REQUEST FUNCTION] Save annotations ERROR: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)

        finally:
            for leftover in (tmp_path, _get_working_file('annotations.bin.patch.tmp')):
                if leftover and os.path.isfile(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass

    return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)


@csrf_exempt
def package_download_view(request):
    """
    Creates a ZIP package containing the selected LAS segments and trained models.
    All temporary files generated during packaging are deleted after the response
    is built (extracted segment .las files). annotations.bin is kept.
    """
    if request.method == 'POST':
        
        # Files to delete after the response is assembled (populated during processing)
        temp_files_to_delete = []

        try:
            print("\n[REQUEST FUNCTION] DOWNLOAD PACKAGE")
            data = json.loads(request.body)
            selected_segments = data.get('segments', [])  # list of {id, label}
            selected_point_cloud_files = data.get('point_cloud_files', [])  # list of {path, label}
            selected_models   = data.get('models', [])    # list of model names
            project_las       = data.get('las_path')      # e.g. viewer/static/.../features.las
            project_bin       = data.get('bin_path')      # annotations.bin (defaults to working/annotations.bin)
            project_name      = data.get('project_name')  # source point cloud base name
            # Convert relative paths to absolute if needed
            if project_las and not os.path.isabs(project_las):
                project_las = os.path.join(settings.BASE_DIR, project_las)
            if not project_bin:
                project_bin = _get_working_file('annotations.bin')
            elif not os.path.isabs(project_bin):
                project_bin = os.path.join(settings.BASE_DIR, project_bin)

            working_dir = _get_working_dir()
            if not project_las:
                project_las = os.path.join(working_dir, 'features.las')

            models_root = _get_models_dir()

            # annotations.bin is the persistent annotation store: never deleted here

            items_to_zip = []  # list of (archive_path, file_content_or_path, is_content)

            # ── 1. Direct point-cloud files (final pipeline outputs) ────────
            # Example: runtime_data/working/<model>/predicted.las
            print("Preparing segments for ZIP package...")
            for file_entry in selected_point_cloud_files:
                raw_path = file_entry.get('path')
                if not raw_path:
                    continue

                abs_path = raw_path
                if not os.path.isabs(abs_path):
                    abs_path = os.path.join(settings.BASE_DIR, abs_path)

                if not os.path.isfile(abs_path):
                    continue

                raw_label = file_entry.get('label') or os.path.basename(abs_path)
                safe_label = raw_label.replace(' ', '_').replace('.', '_')
                if not safe_label.lower().endswith('_las'):
                    safe_label = f"{safe_label}_las"
                safe_label = safe_label.replace('__', '_')
                out_name = safe_label[:-4] + '.las'

                items_to_zip.append((f"segments/{out_name}", abs_path, False))

            # ── 2. Segments (legacy path) ────────────────────────────────────
            # All segments (including 0) go through extract_segment_las.
            # Segment 0 CANNOT be served as the entire features.las:
            # it also contains the points cut into segments 1, 2, ...
            # The C++ tool filters only the points with buffer[pid*2] == seg_id.
            for seg in selected_segments:
                seg_id = int(seg['id'])
                label  = seg['label'].replace(' ', '_').replace('.', '_')
                if not project_bin or not os.path.isfile(project_bin):
                    raise ValueError(
                        f"Annotations bin not found (path: {project_bin}). "
                        "The frontend must save the annotations via /api/export-mapping/ first."
                    )

                seg_las_name = f"segment_{seg_id}.las"
                seg_las_path = os.path.join(working_dir, seg_las_name)

                if not os.path.isfile(seg_las_path):
                    if os.path.isfile(project_las):
                        try:
                            extract_segment_las(project_las, project_bin, seg_id, seg_las_path)
                        except Exception as ex:
                            print(f"[DOWNLOAD] Error extracting segment {seg_id}: {ex}")

                if os.path.isfile(seg_las_path):
                    temp_files_to_delete.append(seg_las_path)
                    items_to_zip.append((f"segments/{label}.las", seg_las_path, False))

            # ── 3. Models ─────────────────────────────────────────────────────
            # Model zip files are built in memory (BytesIO) — no files on disk.
            print("Preparing models for ZIP package...")
            for model_name in selected_models:
                model_dir = os.path.join(models_root, model_name)
                if os.path.isdir(model_dir):
                    model_zip_buf = BytesIO()
                    with zipfile.ZipFile(model_zip_buf, 'w', zipfile.ZIP_DEFLATED) as model_zip:
                        ALLOWED_EXTS = ('.pkl', '.txt', '.json')
                        for root, _, files in os.walk(model_dir):
                            for file in files:
                                if any(file.lower().endswith(ext) for ext in ALLOWED_EXTS):
                                    model_zip.write(os.path.join(root, file), file)
                    model_zip_buf.seek(0)
                    items_to_zip.append((f"models/{model_name}.zip", model_zip_buf.read(), True))

            # ── 4. Build response ─────────────────────────────────────────────
            if len(items_to_zip) == 1 and items_to_zip[0][0].startswith("models/") and items_to_zip[0][2]:
                # Single model: return its zip directly
                content = items_to_zip[0][1]
                model_filename = os.path.basename(items_to_zip[0][0])
                response = HttpResponse(content, content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename="{model_filename}"'
                return response

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for arcname, target, is_content in items_to_zip:
                    if is_content:
                        zip_file.writestr(arcname, target)
                    else:
                        zip_file.write(target, arcname)

            zip_buffer.seek(0)
            response = HttpResponse(zip_buffer.read(), content_type='application/zip')
            safe_project_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', (project_name or '')).strip('_') or 'download'
            response['Content-Disposition'] = f'attachment; filename="{safe_project_name}_package.zip"'
            print(f"Prepared ZIP with {len(items_to_zip)} items ( {len(selected_point_cloud_files)} segments, {len(selected_models)} models )")
            return response

        except Exception as e:
            print(f"[DOWNLOAD ERROR] {e}")
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)

        finally:
            # Delete all temporary files generated during packaging,
            # both in case of success and error.
            for path in temp_files_to_delete:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"[DOWNLOAD] Deleted temp file: {path}")
                except Exception as cleanup_err:
                    print(f"[DOWNLOAD] Warning: could not delete temp file {path}: {cleanup_err}")

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def upload_model(request):
    """
    Endpoint for uploading a trained model in a ZIP file.
    The ZIP must contain the model files (pkl, json, txt).
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] UPLOAD MODEL")
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({"error": "No file provided"}, status=400)
            if not uploaded_file.name.endswith('.zip'):
                return JsonResponse({"error": "Only ZIP files are supported"}, status=400)

            model_name = os.path.splitext(uploaded_file.name)[0]
            models_root = _get_models_dir()
            os.makedirs(models_root, exist_ok=True)
            
            model_dir = os.path.join(models_root, model_name)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_zip_path, 'wb+') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
                
                # Separate directory for extraction to avoid copying the ZIP itself
                extract_path = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_path, exist_ok=True)

                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                    
                    # Logic to handle both flat zip and one-folder-deep zip
                    items = [i for i in os.listdir(extract_path) if not i.startswith('__MACOSX')]
                    extract_target = extract_path
                    if len(items) == 1 and os.path.isdir(os.path.join(extract_path, items[0])):
                        extract_target = os.path.join(extract_path, items[0])

                    # Verify model.pkl
                    if not any(f.endswith('.pkl') for f in os.listdir(extract_target)):
                        return JsonResponse({"error": "ZIP must contain a .pkl model file"}, status=400)

                    # Move content to final folder
                    shutil.copytree(extract_target, model_dir, dirs_exist_ok=True)

            return JsonResponse({"status": "success", "message": f"Model '{model_name}' uploaded successfully", "name": model_name})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)

