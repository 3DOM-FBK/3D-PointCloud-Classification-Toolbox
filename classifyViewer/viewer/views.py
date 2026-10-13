from django.shortcuts import render
from django.conf import settings

# Create your views here.

def main_viewer(request):
    return render(request, "viewer/viewer_page.html", {
        "runtime_data_url": settings.RUNTIME_DATA_URL,
        "runtime_data_path_prefix": settings.RUNTIME_DATA_ROOT.relative_to(settings.BASE_DIR).as_posix(),
        "app_version": settings.APP_VERSION,
    })

def documentation(request):
    return render(request, "viewer/docs_page.html")

