from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, Http404,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .functions import launch_training_RF, launch_classify_RF, subsampling_point_cloud, stop_processes, get_voxel_size, check_point_id, inspect_las_header
from .functions import mesh_to_point_cloud, ply_to_las, feature_extraction, Potree, split_las_by_store, las_to_feature_bin, extract_segment_las
import base64
import os
import json
import struct
import traceback
import re
import datetime
import zipfile
import tempfile
import shutil
from django.conf import settings
from io import BytesIO
from .functions import extract_segment_las


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
def potree_converter(request):
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] POTREE CONVERTER:", request.body[:200]) 
            data = json.loads(request.body)

            input_filepath = data['input_filepath']
            output_filepath = data['output_filepath']
        
            Potree(input_filepath, output_filepath)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Potree conversion completed."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] Potree conversion ERROR " + str(e))
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
    Split LAS point cloud by segment annotations from the .pcbin store.

    POST body (JSON):
        las_path   - path to features.las
        pcbin_path - path to features.pcbin (unified binary store)
        output_dir - destination directory for output segment_*.las files
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] SPLIT LAS BY PCBIN:", request.body[:200])
            data = json.loads(request.body)

            las_path   = data['las_path']
            pcbin_path = data.get('pcbin_path') or data.get('mapping_path')  # backward compat
            output_dir = data['output_dir']
            exclude_unclassified = bool(data.get('exclude_unclassified', False))

            if not pcbin_path:
                raise ValueError("pcbin_path is required")

            split_las_by_store(las_path, pcbin_path, output_dir, exclude_unclassified=exclude_unclassified)

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

@csrf_exempt
def las_to_feature_bin_view(request):
    """
    Generate .pcbin feature store from a LAS file.

    POST body (JSON):
        las_path   - path to features.las
        pcbin_path - destination path for features.pcbin
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] LAS TO FEATURES PCBIN:", request.body[:200])
            data = json.loads(request.body)

            las_path   = data['las_path']
            pcbin_path = data.get('pcbin_path')

            if not pcbin_path:
                raise ValueError("pcbin_path is required")

            las_to_feature_bin(las_path, pcbin_path)
            print("\n")

            return JsonResponse({"status": 'success', "message": "Features pcbin generated successfully."})

        except Exception as e:
            print("\n[REQUEST FUNCTION] LAS TO FEATURES PCBIN ERROR " + str(e))
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
    file, using the .pcbin store for annotation lookup.

    POST body (JSON):
        las_path   - path to features.las
        pcbin_path - path to features.pcbin (unified binary store)
        seg_id     - integer segment ID to extract
        out_path   - destination .las path
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] EXTRACT SEGMENT LAS:", request.body[:200])
            data = json.loads(request.body)

            las_path   = data['las_path']
            pcbin_path = data.get('pcbin_path') or data.get('mapping_path')  # backward compat
            seg_id     = int(data['seg_id'])
            out_path   = data['out_path']

            if not pcbin_path:
                raise ValueError("pcbin_path is required")

            extract_segment_las(las_path, pcbin_path, seg_id, out_path)
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
    Serve binary files (octree.bin, hierarchy.bin) with HTTP Range request support.
    This is critical for Potree 2.0 which needs to fetch small chunks from
    very large files (e.g. 3GB octree.bin).
    
    Without this, Django's dev server returns the entire file for every request,
    which crashes the browser for large files.
    """
    # Security: restrict to specific binary files in the data directory
    ALLOWED_EXTENSIONS = ('.bin', '.json', '.pcbin')
    BASE_DATA_DIR = str(settings.RUNTIME_DATA_ROOT)

    # Normalize and validate path
    # We lstrip('/') to ensure os.path.join doesn't treat it as an absolute path
    full_path = os.path.normpath(os.path.join(BASE_DATA_DIR, filepath.lstrip('/')))

    # Prevent directory traversal
    if not full_path.startswith(os.path.normpath(BASE_DATA_DIR)):
        raise Http404("Access denied")

    if not os.path.isfile(full_path):
        raise Http404(f"File not found: {filepath}")

    _, ext = os.path.splitext(full_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise Http404("File type not allowed")

    file_size = os.path.getsize(full_path)

    # Determine content type
    content_type = 'application/octet-stream'
    if ext.lower() == '.json':
        content_type = 'application/json'

    # Check for Range header
    range_header = request.META.get('HTTP_RANGE', '')

    if range_header:
        # Parse Range: bytes=start-end
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

            # Clamp
            end = min(end, file_size - 1)

            if start > end or start >= file_size:
                response = HttpResponse(status=416)  # Range Not Satisfiable
                response['Content-Range'] = f'bytes */{file_size}'
                return response

            length = end - start + 1

            def file_chunk_iterator():
                with open(full_path, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 65536  # 64KB chunks
                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        data = f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            response = StreamingHttpResponse(
                file_chunk_iterator(),
                status=206,
                content_type=content_type
            )
            response['Content-Length'] = length
            response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response['Accept-Ranges'] = 'bytes'
            response['Access-Control-Allow-Origin'] = '*'
            return response

    # No Range header — serve full file (for small files like metadata.json)
    # For large files, stream it
    if file_size > 10 * 1024 * 1024:  # > 10MB: stream
        def full_file_iterator():
            with open(full_path, 'rb') as f:
                chunk_size = 65536
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        response = StreamingHttpResponse(
            full_file_iterator(),
            content_type=content_type
        )
    else:
        with open(full_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type=content_type)

    response['Content-Length'] = file_size
    response['Accept-Ranges'] = 'bytes'
    response['Access-Control-Allow-Origin'] = '*'
    return response


def serve_runtime_file(request, filepath):
    """Serve runtime data files (LAS, pcbin, JSON) without Range support."""
    ALLOWED_EXTENSIONS = ('.las', '.bin', '.json', '.pcbin', '.txt')
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

    with open(full_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
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
            features_pcbin = _get_working_file('features.pcbin')

            if not os.path.isfile(backup_las):
                return JsonResponse({"error": "point cloud backup not found"}, status=404)

            os.makedirs(working_dir, exist_ok=True)
            shutil.copy2(backup_las, features_las)

            if os.path.isfile(features_pcbin):
                os.remove(features_pcbin)

            las_to_feature_bin(
                _runtime_relative_path('working', 'features.las'),
                _runtime_relative_path('working', 'features.pcbin')
            )

            return JsonResponse({
                "status": "success",
                "message": "Point cloud restored from backup",
                "las_path": _runtime_relative_path('working', 'features.las'),
                "pcbin_path": _runtime_relative_path('working', 'features.pcbin')
            }, status=200)
        except Exception as e:
            print("\n[REQUEST FUNCTION] RESTORE POINT CLOUD BACKUP ERROR " + str(e))
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def start_training(request):
    """
    Endpoint for receiving training data in binary format.
    Expects FormData with:
    - 'labels': JSON string mapping segmentId -> segmentName
    - 'buffer': Binary file (one byte per point total)
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] START TRAINING")
            label_map_json = request.POST.get('labels')
            buffer_file = request.FILES.get('buffer')

            if not label_map_json:
                return JsonResponse({"error": "Missing 'labels' metadata"}, status=400)
            if not buffer_file:
                return JsonResponse({"error": "Missing 'buffer' binary data"}, status=400)

            # Save in the working directory used by the pipeline
            training_dir = _get_working_dir()
            os.makedirs(training_dir, exist_ok=True)

            # abs_input = os.path.abspath(os.path.join(settings.BASE_DIR, input_filepath))

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bin_filename = f"labels_{timestamp}.bin"
            json_filename = f"meta_{timestamp}.json"

            # Save binary buffer
            bin_path = os.path.join(training_dir, bin_filename)
            with open(bin_path, 'wb') as f:
                for chunk in buffer_file.chunks():
                    f.write(chunk)
            # Save metadata JSON
            json_path = os.path.join(training_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(label_map_json)

            return JsonResponse({
                "message": "Binary data saved successfully",
                "filename": bin_filename,
                "labels_filename": json_filename,
                "bin_path": bin_path,
                "json_path": json_path,
                "size_bytes": os.path.getsize(bin_path)
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)


@csrf_exempt
def export_mapping(request):
    """
    Update segmentation annotations inside the unified .pcbin store.

    Reads the existing features.pcbin, patches segment_id and manual_class_id
    for each annotated point (where buffer segment_id != 0), then writes the
    file back atomically via a temp file + rename.

    Expects FormData with:
    - 'buffer'      : Binary blob (2 bytes per point: segment_id, class_id)
    - 'point_count' : Integer (total number of points in the buffer)
    - 'pcbin_path'  : Relative path to features.pcbin (defaults to
                      'runtime_data/working/features.pcbin')

    Returns JSON:
    - pcbin_path    : same relative path that was updated
    - point_count   : number of annotated points written
    """
    if request.method == 'POST':
        try:
            print("\n[REQUEST FUNCTION] EXPORT MAPPING (pcbin update)")
            buffer_file = request.FILES.get('buffer')
            point_count_str = request.POST.get('point_count')
            pcbin_path = request.POST.get(
                'pcbin_path',
                _runtime_relative_path('working', 'features.pcbin')
            )

            if not buffer_file:
                return JsonResponse({"error": "Missing 'buffer' binary data"}, status=400)
            if not point_count_str:
                return JsonResponse({"error": "Missing 'point_count'"}, status=400)

            try:
                point_count = int(point_count_str)
            except ValueError:
                return JsonResponse({"error": "Invalid point_count (must be integer)"}, status=400)

            # Read binary buffer
            buffer_data = b''.join(buffer_file.chunks())

            if len(buffer_data) != point_count * 2:
                return JsonResponse({
                    "error": f"Buffer size mismatch: expected {point_count * 2} bytes, got {len(buffer_data)}"
                }, status=400)

            # Resolve absolute path (pcbin_path is relative to BASE_DIR)
            abs_pcbin = os.path.normpath(os.path.join(settings.BASE_DIR, pcbin_path))

            # Security: must stay inside BASE_DIR
            if not abs_pcbin.startswith(os.path.normpath(settings.BASE_DIR)):
                return JsonResponse({"error": "Invalid pcbin_path"}, status=400)

            if not os.path.isfile(abs_pcbin):
                # features.pcbin doesn't exist yet (e.g. cloud was loaded without running
                # feature calculation or training). Generate it from features.las first.
                las_relative = _runtime_relative_path('working', 'features.las')
                abs_las = os.path.normpath(os.path.join(settings.BASE_DIR, las_relative))
                if not os.path.isfile(abs_las):
                    return JsonResponse({"error": "features.las not found — please load a point cloud first"}, status=404)
                print(f"[REQUEST FUNCTION] features.pcbin not found — generating from {las_relative}")
                las_to_feature_bin(las_relative, pcbin_path)
                print("[REQUEST FUNCTION] features.pcbin generated")

            # Read existing pcbin into mutable bytearray
            with open(abs_pcbin, 'rb') as f:
                raw = f.read()

            # Validate magic
            if raw[:4] != b'PCBN':
                return JsonResponse({"error": "Invalid pcbin file (bad magic)"}, status=400)

            version = raw[4]
            if version == 1:
                bpf = 4
            elif version == 2:
                bpf = raw[5]   # bytes_per_feature stored at header byte [5]
                if bpf not in (1, 4):
                    return JsonResponse({"error": f"Unsupported pcbin bpf={bpf}"}, status=400)
            else:
                return JsonResponse({"error": f"Unsupported pcbin version={version}"}, status=400)

            N = struct.unpack_from('<I', raw, 8)[0]   # number of point slots
            F = struct.unpack_from('<I', raw, 12)[0]  # number of features
            header_size = 16 + F * 40                 # magic/ver/bpf/reserved/N/F + F*(name32+vmin+vmax)
            record_size = F * bpf + 8                 # F×bpf feature bytes + 4 annotation + 4 confidence

            data = bytearray(raw)

            annotated = 0
            for pid in range(min(point_count, N)):
                seg_id_buf = buffer_data[pid * 2]      # 1-based in buffer (0 = unannotated)
                class_id   = buffer_data[pid * 2 + 1]
                if seg_id_buf != 0:
                    rec_off = header_size + pid * record_size
                    data[rec_off + F * bpf]     = seg_id_buf - 1         # restore 0-based segment_id
                    data[rec_off + F * bpf + 1] = class_id if class_id != 0 else 0xFF  # 0 = JS "no class" → 0xFF sentinel
                    annotated += 1

            # Atomic write
            tmp_path = abs_pcbin + '.tmp'
            with open(tmp_path, 'wb') as f:
                f.write(data)
            os.replace(tmp_path, abs_pcbin)

            print(f"[REQUEST FUNCTION] pcbin updated: {annotated} annotated points → {abs_pcbin}")

            return JsonResponse({
                "pcbin_path": pcbin_path,
                "point_count": annotated,
            }, status=200)

        except Exception as e:
            print(f"[REQUEST FUNCTION] Export mapping ERROR: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)


@csrf_exempt
def package_download_view(request):
    """
    Creates a ZIP package containing the selected LAS segments and trained models.
    All temporary files generated during packaging are deleted after the response
    is built (segment .las, segment .bin, metadata .json).
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
            project_bin       = data.get('bin_path')      # labels_TIMESTAMP.bin from /api/start-training/
            project_name      = data.get('project_name')  # source point cloud base name
            # Convert relative paths to absolute if needed
            if project_las and not os.path.isabs(project_las):
                project_las = os.path.join(settings.BASE_DIR, project_las)
            # project_bin arrives already absolute from /api/start-training/, handle
            # the relative case as well for robustness

            if project_bin and not os.path.isabs(project_bin):
                project_bin = os.path.join(settings.BASE_DIR, project_bin)

            working_dir = _get_working_dir()
            if not project_las:
                project_las = os.path.join(working_dir, 'features.las')

            models_root = _get_models_dir()

            # Mark the bin and json generated by /api/start-training/ for cleanup
            # The bin is named labels_TIMESTAMP.bin → the associated json is meta_TIMESTAMP.json
            if project_bin and os.path.isfile(project_bin):
                temp_files_to_delete.append(project_bin)
                # The json has the same timestamp: labels_20240101_120000.bin → meta_20240101_120000.json
                bin_basename = os.path.basename(project_bin)  # labels_TIMESTAMP.bin
                if bin_basename.startswith('labels_'):
                    json_name = bin_basename.replace('labels_', 'meta_', 1).replace('.bin', '.json')
                    json_path = os.path.join(os.path.dirname(project_bin), json_name)
                    if os.path.isfile(json_path):
                        temp_files_to_delete.append(json_path)

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
                        f"Segment bin not found (path: {project_bin}). "
                        "The frontend must send the segment buffer via /api/start-training/ first."
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

