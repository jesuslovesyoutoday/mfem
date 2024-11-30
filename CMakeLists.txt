CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

project(project1)

#set(MFEM_DIR "/home/scandiy/lib/mfem-4.5.2/build")
#find_package(mfem REQUIRED PATHES "${MFEM_DIR}" NO_DEFAULT_PATH)
#find_package(mfem REQUIRED)


#include_directories("/home/scandiy/lib/mfem-4.5.2/build")

add_executable(project1 main.cpp)


#find_package(MFEM REQUIRED)

#message(">>>>>>>>>>>>>  "  ${MFEM_LIBRARIES})

#set(BLA_VENDOR OpenBLAS)

#find_package(OpenBLAS) 

#message(STATUS "$$$$$$$$$$$$$$$$$$$$$$   " ${BLAS_INCLUDE_DIR})


target_include_directories(project1 PRIVATE /usr/local/include)

target_link_libraries(project1 PRIVATE /usr/local/lib/libmfem.a)


#target_link_libraries(project1 PRIVATE /usr/local/lib/libamd.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libbtf.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libcolamd.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libcholmod.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libccolamd.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libcamd.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libklu.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libumfpack.a)
#target_link_libraries(project1 PRIVATE /usr/local/lib/libsuitesparseconfig.a)
#target_link_libraries(project1 PRIVATE /lib/x86_64-linux-gnu/libopenblas.a)






