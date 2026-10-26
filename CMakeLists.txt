cmake_minimum_required(VERSION 3.18)

option(SMINA_GPU "Build the CUDA-accelerated code path (enabled with --gpu at runtime)" ON)

if(SMINA_GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Ada (both development GPUs) plus PTX so newer GPUs can JIT
    set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual")
  endif()
  project(smina LANGUAGES CXX CUDA)
else()
  project(smina LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# match the original makefiles: -O3 -DNDEBUG -g
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -lineinfo")

find_package(Boost 1.70 REQUIRED COMPONENTS
  program_options thread system filesystem serialization iostreams timer regex date_time)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

find_path(OPENBABEL_INCLUDE_DIR openbabel/mol.h
  PATH_SUFFIXES openbabel3 openbabel-2.0)
find_library(OPENBABEL_LIBRARY NAMES openbabel)
if(NOT OPENBABEL_INCLUDE_DIR OR NOT OPENBABEL_LIBRARY)
  message(FATAL_ERROR "OpenBabel not found")
endif()

set(LIB_SOURCES
  builtinscoring.cpp cache.cpp coords.cpp custom_terms.cpp everything.cpp
  flexinfo.cpp grid.cpp szv_grid.cpp model.cpp monte_carlo.cpp mutate.cpp
  my_pid.cpp naive_non_cache.cpp non_cache.cpp obmolopener.cpp parallel_mc.cpp
  parse_pdbqt.cpp pdb.cpp quasi_newton.cpp quaternion.cpp random.cpp ssd.cpp
  terms.cpp weighted_terms.cpp molgetter.cpp result_info.cpp
  PDBQTUtilities.cpp SminaConverter.cpp)
list(TRANSFORM LIB_SOURCES PREPEND ${CMAKE_SOURCE_DIR}/src/lib/)

add_library(sminalib STATIC ${LIB_SOURCES})
target_include_directories(sminalib PUBLIC
  ${CMAKE_SOURCE_DIR}/src/lib ${OPENBABEL_INCLUDE_DIR})
set(SMINA_LIB_DEPS
  ${OPENBABEL_LIBRARY} Boost::program_options Boost::thread Boost::system
  Boost::filesystem Boost::serialization Boost::iostreams Boost::timer
  Boost::regex Boost::date_time ZLIB::ZLIB Threads::Threads)
target_link_libraries(sminalib PUBLIC ${SMINA_LIB_DEPS})
# boost/progress.hpp is deprecated but still what smina uses
target_compile_definitions(sminalib PUBLIC BOOST_TIMER_ENABLE_DEPRECATED
  BOOST_ALLOW_DEPRECATED_HEADERS)

add_executable(smina ${CMAKE_SOURCE_DIR}/src/main/main.cpp)
target_link_libraries(smina PRIVATE sminalib)


if(SMINA_GPU)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/gpu)
  target_compile_definitions(smina PRIVATE SMINA_GPU)
  target_link_libraries(smina PRIVATE sminagpu)
endif()
