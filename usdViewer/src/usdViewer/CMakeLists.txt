add_executable (usdViewer
        main.cpp
        AnimatedMesh.h
        )

find_package(PythonLibs)
find_package(Eigen3)

set(EIGEN3_INC_DIR /u/s/g/sgsrinivasa2/eigen-3.3.7) 
set(PYTHON_INCLUDE_PATH /usr/include/python3.6m)

target_include_directories(usdViewer
        PUBLIC
            ${USD_INC_DIR}
            ${EIGEN3_INC_DIR}
            ${PYTHON_INCLUDE_PATH}
        )

target_link_libraries(
        usdViewer
                ${USD_LIB_DIR}/libgf.so
                ${USD_LIB_DIR}/libsdf.so
                ${USD_LIB_DIR}/libtf.so
                ${USD_LIB_DIR}/libusd.so
                ${USD_LIB_DIR}/libusdGeom.so
                ${USD_LIB_DIR}/libvt.so
                ${USD_LIB_DIR}/libboost_python.so # todo find library
                ${PYTHON_LIBRARY}
        )
