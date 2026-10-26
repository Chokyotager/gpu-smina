# device code (independent of the smina CPU library)
add_library(sminagpu_kernels STATIC
  gpu_device.cu
  gpu_kernels_f32.cu
  gpu_kernels_f64.cu)
target_include_directories(sminagpu_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/lib)
target_link_libraries(sminagpu_kernels PUBLIC CUDA::cudart)

# host glue, against the production CPU library
add_library(sminagpu STATIC gpu_flatten.cpp gpu_search.cpp gpu_docking.cpp)
target_link_libraries(sminagpu PUBLIC sminalib sminagpu_kernels)

# validation variant: CPU library with portable_math transcendental functions
# so that CPU and double precision GPU trajectories match bit for bit
add_library(sminalib_portable STATIC ${LIB_SOURCES})
target_include_directories(sminalib_portable PUBLIC
  ${CMAKE_SOURCE_DIR}/src/lib ${OPENBABEL_INCLUDE_DIR})
target_link_libraries(sminalib_portable PUBLIC ${SMINA_LIB_DEPS})
target_compile_definitions(sminalib_portable PUBLIC SMINA_PORTABLE_MATH
  BOOST_TIMER_ENABLE_DEPRECATED BOOST_ALLOW_DEPRECATED_HEADERS)
add_library(sminagpu_portable STATIC gpu_flatten.cpp gpu_search.cpp)
target_link_libraries(sminagpu_portable PUBLIC sminalib_portable sminagpu_kernels)

# Double precision is the validation path: keep CPU-identical rounding by
# disabling FMA contraction (x86-64 smina is built without FMA).
set_source_files_properties(gpu_kernels_f64.cu gpu_device.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")
target_compile_options(sminagpu_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wno-deprecated-declarations>)

add_executable(gpu_eval_test tests/gpu_eval_test.cpp)
target_link_libraries(gpu_eval_test PRIVATE sminagpu)

add_executable(gpu_mc_test tests/gpu_mc_test.cpp)
target_link_libraries(gpu_mc_test PRIVATE sminagpu_portable)
