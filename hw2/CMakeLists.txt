cmake_minimum_required(VERSION 3.14)
project(hw2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(OpenMP REQUIRED)

# Group number
set(GROUP_NO 68 CACHE STRING "Your group number (two digits)")

# Packaging system
if (NOT ${GROUP_NO} STREQUAL 00)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "CS5220Group${GROUP_NO}_hw2")
    install(FILES serial.cpp openmp.cpp DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(serial main.cpp serial.cpp common.h)

add_executable(openmp main.cpp openmp.cpp common.h)
target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
configure_file(job-openmp job-openmp COPYONLY)
