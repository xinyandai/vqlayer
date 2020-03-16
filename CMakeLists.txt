cmake_minimum_required(VERSION 2.8)
project(vqlayer)

#SET( CMAKE_CXX_FLAGS
#        "${CMAKE_CXX_FLAGS} -Ofast -lrt -lm -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -mtune=generic -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
SET( CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -lm -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -mtune=generic -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

find_package( PkgConfig )

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#find_package( BLAS REQUIRED )
#message( STATUS BLAS found: ${BLAS_LIBRARIES} )
#link_libraries(-lpthread ${BLAS_LIBRARIES})
file(GLOB_RECURSE VQ_LAYER_SOURCES "src/*.cc")
add_executable(main app/main.cc ${VQ_LAYER_SOURCES})

set(test_set  smm vq rqlayer pqlayer selector)
foreach(test  ${test_set})
    add_executable(test_${test} test/test_${test}.cc ${VQ_LAYER_SOURCES})
endforeach()