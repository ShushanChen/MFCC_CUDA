cmake_minimum_required (VERSION 2.8)
project (Speech)
find_package(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)

set (PRO2_MAIN pro2_demo)
add_executable(${PRO2_MAIN} ${PRO2_MAIN}.cpp)

set (PRO2_CUDA_MAIN pro2_cuda_demo)
cuda_add_executable(${PRO2_CUDA_MAIN} ${PRO2_MAIN}.cpp)

set (RECOREDER recorder)
add_executable(${RECOREDER} ${RECOREDER}.cpp)

set(ANALYSIS Analysis)
set(CAPTURE Capture)
set(FEATURE Feature)
set(TEST test)
set(CONFIGURE Configure)
set(READWAVE readwave)
set(DATA data)
set(THREADPOOL ThreadPool)
set(MFCC_CUDA MfccCuda)

ADD_DEFINITIONS(
    -std=c++11
)

cuda_include_directories ("${PROJECT_SOURCE_DIR}/${TEST}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${FEATURE}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${CAPTURE}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${ANALYSIS}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${CONFIGURE}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${READWAVE}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${DATA}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/${THREADPOOL}")

include_directories ("${PROJECT_SOURCE_DIR}/${TEST}")
include_directories ("${PROJECT_SOURCE_DIR}/${FEATURE}")
include_directories ("${PROJECT_SOURCE_DIR}/${CAPTURE}")
include_directories ("${PROJECT_SOURCE_DIR}/${ANALYSIS}")
include_directories ("${PROJECT_SOURCE_DIR}/${CONFIGURE}")
include_directories ("${PROJECT_SOURCE_DIR}/${READWAVE}")
include_directories ("${PROJECT_SOURCE_DIR}/${DATA}")
include_directories ("${PROJECT_SOURCE_DIR}/${THREADPOOL}")

add_subdirectory (${TEST})
add_subdirectory (${FEATURE})
add_subdirectory (${CAPTURE})
add_subdirectory (${ANALYSIS})
add_subdirectory (${CONFIGURE})
add_subdirectory (${READWAVE})
add_subdirectory (${DATA})
add_subdirectory (${THREADPOOL})


target_link_libraries(${PRO2_CUDA_MAIN} ${MFCC_CUDA})
target_link_libraries(${PRO2_CUDA_MAIN} ${TEST})
target_link_libraries(${PRO2_CUDA_MAIN} ${CAPTURE})
target_link_libraries(${PRO2_CUDA_MAIN} ${ANALYSIS})
target_link_libraries(${PRO2_CUDA_MAIN} ${DATA})
target_link_libraries(${PRO2_CUDA_MAIN} ${CONFIGURE})
target_link_libraries(${PRO2_CUDA_MAIN} ${THREADPOOL})
target_link_libraries(${PRO2_CUDA_MAIN} ${READWAVE})
target_link_libraries(${PRO2_CUDA_MAIN} portaudio)
target_link_libraries(${PRO2_CUDA_MAIN} pthread)

target_link_libraries(${PRO2_MAIN} ${FEATURE})
target_link_libraries(${PRO2_MAIN} ${TEST})
target_link_libraries(${PRO2_MAIN} ${CAPTURE})
target_link_libraries(${PRO2_MAIN} ${ANALYSIS})
target_link_libraries(${PRO2_MAIN} ${DATA})
target_link_libraries(${PRO2_MAIN} ${CONFIGURE})
target_link_libraries(${PRO2_MAIN} ${THREADPOOL})
target_link_libraries(${PRO2_MAIN} ${READWAVE})
target_link_libraries(${PRO2_MAIN} portaudio)
target_link_libraries(${PRO2_MAIN} pthread)

target_link_libraries(${RECOREDER} ${CAPTURE})
target_link_libraries(${RECOREDER} ${ANALYSIS})
target_link_libraries(${RECOREDER} ${DATA})
target_link_libraries(${RECOREDER} ${CONFIGURE})
target_link_libraries(${RECOREDER} ${READWAVE})
target_link_libraries(${RECOREDER} portaudio)
target_link_libraries(${RECOREDER} pthread)


