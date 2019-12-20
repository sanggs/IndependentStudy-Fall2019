add_executable (usdViewer
        main.cpp
        AnimatedMesh.h
        )

find_package(PythonLibs)
find_package(Eigen3)

set(EIGEN3_INC_DIR /usr/local/include/eigen3) 
set(PYTHON_INCLUDE_PATH /Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m)

target_include_directories(usdViewer
        PUBLIC
            ${USD_INC_DIR}
            ${EIGEN3_INC_DIR}
            ${PYTHON_INCLUDE_PATH}
        )

target_link_libraries(
        usdViewer
                ${USD_LIB_DIR}/libgf.dylib
                ${USD_LIB_DIR}/libsdf.dylib
                ${USD_LIB_DIR}/libtf.dylib
                ${USD_LIB_DIR}/libusd.dylib
                ${USD_LIB_DIR}/libusdGeom.dylib
                ${USD_LIB_DIR}/libvt.dylib
                ${USD_LIB_DIR}/libboost_python.dylib # todo find library
                ${PYTHON_LIBRARY}
        )
