set(CMAKE_CUDA_ARCHITECTURES 80 90)
set(target gda)
set(efa "${PROJECT_SOURCE_DIR}/modules/efa-dp-direct/CUDA/src")
file(GLOB src *.cu *.cc "${efa}/*.cpp" "${efa}/*.cu")

add_executable(${target} ${src})
set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${target} PRIVATE -Wall -O3 -g)
target_include_directories(${target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${HWLOC_INCLUDE_DIRS}"
  "${MPI_INCLUDE_PATH}"
  "${efa}"
  "/opt/gdrcopy/include"
)
target_link_libraries(${target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  spdlog::spdlog
  "${MPI_LIBRARIES}"
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
  "${GDR_LIBRARY}"
)
