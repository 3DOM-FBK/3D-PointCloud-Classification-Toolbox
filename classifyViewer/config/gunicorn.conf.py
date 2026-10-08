"""
Gunicorn Configuration for 3D Classify Viewer
==============================================

This file contains the configuration settings for the Gunicorn server
that hosts the Django 3D Classify Viewer application.
"""

# Server binding address and port
# 0.0.0.0 allows connections from any IP address
# 8000 is the port on which the server will listen
bind = "0.0.0.0:8000"

# Number of worker processes handling requests
# MUST stay 1: functions.job (JobManager) is a per-process singleton that owns the running
# subprocess, so /stop_process/ has to reach the same process that started the job.
workers = 1

# Number of threads per worker
# Calculation endpoints (feature extraction, RF, conversion) are synchronous and keep a thread busy
# for the whole job, while the viewer streams the point cloud with up to 6 parallel Range requests
# (Chrome's per-host limit). 8 threads keep the file serving responsive during long jobs.
threads = 8

# Request timeout in seconds (0 = disabled)
# Useful for classification operations that may take a long time
timeout = 0
