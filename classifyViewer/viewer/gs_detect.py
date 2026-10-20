"""Detects Gaussian splat PLY files from the header alone (no numpy, no full read).

A file is a Gaussian splat when its vertex element has the 3DGS/2DGS properties: f_dc_0..2, opacity,
scale_0 and rot_0. Returns '3dgs', '2dgs' (two scales) or None.
"""

GS_MESSAGE = (
    "Il file è un PLY Gaussian Splat (3DGS/2DGS). L'import come nuvola di punti non è supportato: "
    "i dati dello splat (colore SH, opacità, scala, rotazione) non verrebbero letti. "
    "Il supporto GS nel viewer principale non è ancora implementato: usa il convertitore "
    "classifyViewer/viewer/utils_functions/gs_convert.py e il renderer di prova testC++/gs/viewer."
)


def gaussian_kind(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(1 << 16)
    except OSError:
        return None
    end = head.find(b'end_header')
    if not head.startswith(b'ply') or end < 0:
        return None
    props = []
    in_vertex = False
    for line in head[:end].decode('ascii', 'replace').splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'element':
            if in_vertex:
                break                      # only the vertex element is inspected
            in_vertex = parts[1] == 'vertex'
        elif parts[0] == 'property' and in_vertex and parts[1] != 'list':
            props.append(parts[-1])
    names = set(props)
    if not {'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'rot_0'} <= names:
        return None
    return '2dgs' if 'scale_2' not in names else '3dgs'
