add_library(sminagpu STATIC
  gpu_flatten.cpp
  gpu_device.cu
  gpu_kernels_f32.cu
  gpu_kernels_f64.cu)
target_include_directories(sminagpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sminagpu PUBLIC sminalib CUDA::cudart)
set_target_properties(sminagpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Double precision is the validation path: keep CPU-identical rounding by
# disabling FMA contraction (x86-64 smina is built without FMA).
set_source_files_properties(gpu_kernels_f64.cu gpu_device.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")
target_compile_options(sminagpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wno-deprecated-declarations>)

add_executable(gpu_eval_test tests/gpu_eval_test.cpp)
target_link_libraries(gpu_eval_test PRIVATE sminagpu)
