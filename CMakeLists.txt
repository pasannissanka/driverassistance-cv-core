cmake_minimum_required(VERSION 3.0.0)
project(yolo VERSION 0.1.0)

include(CTest)
enable_testing()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
include(CPack)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn/include)

add_library( ncnn STATIC IMPORTED )
set_target_properties( # Specifies the target library.
        ncnn
        # Specifies the parameter you want to define.
        PROPERTIES IMPORTED_LOCATION
        # Provides the path to the library you want to import.
        ${CMAKE_SOURCE_DIR}/include/ncnn/lib/libncnn.a )

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(yolo ${SOURCE_FILES})

target_link_libraries( yolo ${OpenCV_LIBS} ${Eigen_LIBRARIES} ncnn -fopenmp)

