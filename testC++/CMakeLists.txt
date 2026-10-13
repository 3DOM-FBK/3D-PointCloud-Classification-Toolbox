cmake_minimum_required(VERSION 3.18)
project(point_cloud_tools)

# --- C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
# CUDA 12.8 or newer is required for the sm_100/sm_120 (Blackwell) targets; "120" (no -real) also embeds PTX.
# No -march=native: the binaries are distributed and must run on any x86-64 CPU. Hot loops pick their
# instruction set at run time (target_clones, see ooc/simd_dispatch.h).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# --- Dependencies ---
# (PCL is no longer needed: the feature extractor has its own neighbour search)
find_package(PDAL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Open3D REQUIRED)
find_package(CGAL REQUIRED)

# --- tinygltf / stb ---
set(TINYGLTF_INCLUDE_DIR /app/tinygltf)

# --- LASzip & laz-perf (manual includes) ---
# Se hai installato in /usr/local, puoi fare così:
include_directories(/usr/local/include /usr/include)
link_directories(/usr/local/lib)

# --- Common Flags ---
add_compile_options(-O3)
add_definitions(${PDAL_DEFINITIONS})

# --- Open3D RPATH ---
get_target_property(open3d_lib Open3D::Open3D IMPORTED_LOCATION)
get_filename_component(open3d_lib_dir ${open3d_lib} DIRECTORY)

# =======================================
# Executables
# =======================================

# --- Feature Extraction Viewer: one program, GPU (CUDA) and CPU backends, no PCL ---
# The backend is chosen at run time (--backend auto|gpu|cpu): the GPU when a usable CUDA device exists, else the CPU.
# The CPU code picks SSE2/AVX2/AVX-512 at run time (target_clones), the GPU code carries SASS for Pascal..Blackwell + PTX.
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(feature_extraction_viewer
    feature_extraction_viewer.cpp
    cpu_features.cpp
    gpu_features.cu
)
target_include_directories(feature_extraction_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(feature_extraction_viewer PRIVATE FEATURES_WITH_CUDA)
set_target_properties(feature_extraction_viewer PROPERTIES
    CUDA_ARCHITECTURES "61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120"
)
target_link_libraries(feature_extraction_viewer
    PRIVATE
        CUDA::cudart_static
        OpenMP::OpenMP_CXX
)
target_compile_options(feature_extraction_viewer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
)

# --- Subsample PC ---
add_executable(subsample_pc subsample_pc.cpp)
target_include_directories(subsample_pc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(subsample_pc PRIVATE Open3D::Open3D OpenMP::OpenMP_CXX)
set_target_properties(subsample_pc PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${open3d_lib_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# --- Mesh2PC ---
add_executable(mesh2pc mesh2pc.cpp)
target_include_directories(mesh2pc PRIVATE ${TINYGLTF_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mesh2pc PRIVATE Open3D::Open3D CGAL::CGAL OpenMP::OpenMP_CXX)
set_target_properties(mesh2pc PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${open3d_lib_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# --- Ply2LAS ---
add_executable(ply2las ply2las.cpp)
target_include_directories(ply2las PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ply2las PRIVATE Open3D::Open3D OpenMP::OpenMP_CXX)
set_target_properties(ply2las PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${open3d_lib_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# --- Split LAS by Binary (streaming, no PDAL: only the standard library) ---
add_executable(split_las_by_binary split_las_by_binary.cpp)
target_include_directories(split_las_by_binary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# --- Check POINT ID (streaming, no Open3D: only the standard library and OpenMP) ---
add_executable(check_point_id check_point_id.cpp)
target_include_directories(check_point_id PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(check_point_id PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(check_point_id PRIVATE ${OpenMP_CXX_FLAGS})

# --- LAS -> chunked point cloud (no external dependencies besides OpenMP) ---
add_executable(las2pc las2pc.cpp)
target_link_libraries(las2pc PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(las2pc PRIVATE ${OpenMP_CXX_FLAGS})
