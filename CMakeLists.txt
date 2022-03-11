cmake_minimum_required(VERSION 3.0.0)
project(yolo VERSION 0.1.0)

include(CTest)
enable_testing()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(ncnn/include)
add_library( ncnn STATIC IMPORTED )
set_target_properties( # Specifies the target library.
        ncnn
        # Specifies the parameter you want to define.
        PROPERTIES IMPORTED_LOCATION
        # Provides the path to the library you want to import.
        ${CMAKE_SOURCE_DIR}/ncnn/lib/libncnn.a )

add_executable(yolo main.cpp yolov4.cpp)
target_link_libraries( yolo ${OpenCV_LIBS} ncnn -fopenmp)

