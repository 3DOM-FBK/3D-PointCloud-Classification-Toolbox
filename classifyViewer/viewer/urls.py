from django.urls import path
from .views import main_viewer, documentation
from .request_functions import save_file, launch_RF_training, launch_RF_classify, subsample_pc, stop_process, get_model_voxel_size, checking_point_id, inspect_las_input, job_progress_view
from .request_functions import mesh2pc, ply2las, feat_extraction, build_pointcloud_view, update_pointcloud_columns_view, _split_las_by_binary, read_text_file, model_exists, models_list, delete_model, extract_segment_las_view
from .request_functions import serve_range_file, serve_runtime_file, clear_data, upload_data, export_mapping, package_download_view, upload_model, backup_pointcloud, restore_pointcloud_backup

urlpatterns = [
    path('', main_viewer, name='main_viewer'),
    path('documentation/', documentation, name='documentation'),
    # Serve point cloud binary files with Range request support
    path('pointcloud-data/<path:filepath>', serve_range_file, name='serve_range_file'),
    # Serve runtime data files (LAS, JSON, etc.)
    path('runtime-data/<path:filepath>', serve_runtime_file, name='serve_runtime_file'),
    path("save_file/", save_file),
    path("launch_RF_training/", launch_RF_training),
    path("launch_RF_classify/", launch_RF_classify),
    path("api/clear-data/", clear_data),
    path("api/upload-data/", upload_data),
    path("api/backup-pointcloud/", backup_pointcloud),
    path("api/restore-pointcloud-backup/", restore_pointcloud_backup),
    path("api/export-mapping/", export_mapping),
    path("subsample_pc/", subsample_pc),
    path("get_model_voxel_size/", get_model_voxel_size),
    path("mesh2pc/", mesh2pc),
    path("ply2las/", ply2las),
    path("check_point_id/", checking_point_id),
    path("inspect_las_input/", inspect_las_input),
    path("feature_extraction/", feat_extraction), 
    path("api/build-pointcloud/", build_pointcloud_view),
    path("api/update-pointcloud-columns/", update_pointcloud_columns_view),
    path("split_las_by_binary/", _split_las_by_binary),
    path("stop_process/", stop_process),
    path("api/job-progress/", job_progress_view),
    path('api/read-file/', read_text_file, name='read_text_file'),
    path('api/model-exists/', model_exists, name='model_exists'),
    path('api/models-list/', models_list, name='models_list'),
    path('api/delete-model/', delete_model, name='delete_model'),
    path('api/extract-segment-las/', extract_segment_las_view, name='extract_segment_las'),
    path('api/download-package/', package_download_view, name='package_download'),
    path("api/upload-model/", upload_model),
]