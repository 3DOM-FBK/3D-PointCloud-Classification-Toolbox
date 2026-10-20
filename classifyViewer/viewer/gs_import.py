"""Import of Gaussian splat files into the viewer: converts them to the `gs` v2 folder.

Input: a PLY that has the 3DGS/2DGS properties, or a SOG v2 file. Anything else is reported as a point cloud
and goes through the normal import path.
Output folder: <RUNTIME_DATA_ROOT>/working/gs (served under /runtime-data/working/gs).
"""
import os
import shutil
import sys

from django.conf import settings

from .gs_detect import gaussian_kind

GS_DIR = 'gs'
UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils_functions')


def splat_kind(abs_path):
    """'gs' for a splat file (PLY with splat properties or SOG), None for anything else."""
    lower = abs_path.lower()
    if lower.endswith('.sog'):
        return 'sog'
    if lower.endswith('.ply') and gaussian_kind(abs_path):
        return 'gs'
    return None


def convert_to_working(abs_path):
    """Converts a splat file into the working gs folder (replacing the previous one). Returns the folder."""
    # the converter lives in utils_functions as plain modules (like pc_columns): imported here, on use only
    if UTILS_DIR not in sys.path:
        sys.path.insert(0, UTILS_DIR)
    import gs_format
    import gs_io

    gs = gs_io.read_source(abs_path)
    summary = gs_io.describe(gs)
    if summary['nonfinite']:
        raise ValueError(f'{summary["nonfinite"]} non-finite values in the splat file')
    out_dir = os.path.join(str(settings.RUNTIME_DATA_ROOT), 'working', GS_DIR)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    gs_format.write_gs(gs, out_dir)
    print(f'[GS IMPORT] {abs_path}: {len(gs)} splats ({gs.kind}, SH degree {gs.sh_degree}) -> {out_dir}')
    return out_dir
