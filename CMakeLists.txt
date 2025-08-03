cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use strict C++ standard
# Ensure MPI is found before including MFEM
# If serial, comment out the following line
project(MyApp)
find_package(MPI REQUIRED)

# This will be the core library for this project
add_library(core INTERFACE)

##---------------------------------------------------------------
# Modify below block with your custom libraries and executables
##---------------------------------------------------------------
# put your non-executable libraries (relative to ./src/) here.
set(LIB
  logger
)

# if a library depends on other libraries, put them like this:
# The dependent libraries should be defined before the library that depends on them
# set(LIB_DEP_mylib dependent_lib1 dependent_lib2)

# put your executables (relative to ./) here.
set(EXECUTABLES
  ex0
  ex1
  ex2
)
# Again, if an executable depends on a specific library, put them like this:
# set(EXE_DEP_myexe dependent_lib1 dependent_lib2)

##---------------------------------------------------------------

#--------------------------------------------------------------
# 1. Setup build type (defaults to release)
if (NOT DEFINED CMAKE_BUILD_TYPE)
  # default build type is Release
  set(CMAKE_BUILD_TYPE "Release")
endif()

#--------------------------------------------------------------
# 2. MFEM configuration
# By default, we assume mfem is in the parent directory of this project with name "mfem"
# You can specify the MFEM build directories in a user.cmake file
# -- file: user.cmake--
# | set(MFEM_RELEASE_DIR "/path/to/mfem/build")
# | set(MFEM_DEBUG_DIR "/path/to/mfem/build-debug")
# -- end user.cmake --
# NOTE: MFEM Should be built with CMAKE
if(EXISTS "${CMAKE_SOURCE_DIR}/user.cmake")
  include("${CMAKE_SOURCE_DIR}/user.cmake")
else()
    # use default if user.cmake does not exist
    set(MFEM_RELEASE_DIR "${CMAKE_SOURCE_DIR}/../mfem/build")
    set(MFEM_DEBUG_DIR "${CMAKE_SOURCE_DIR}/../mfem/build-debug") 
endif()
# Check if MFEM_RELEASE_DIR and MFEM_DEBUG_DIR are set
if (NOT DEFINED MFEM_RELEASE_DIR OR NOT DEFINED MFEM_DEBUG_DIR)
  message(FATAL_ERROR "MFEM_RELEASE_DIR and MFEM_DEBUG_DIR must be set in user.cmake.")
endif()
# Determine the MFEM directory based on the build type
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
if (_build_type STREQUAL "DEBUG")
  set(MFEM_DIR "${MFEM_DEBUG_DIR}")
elseif (_build_type STREQUAL "RELEASE")
  set(MFEM_DIR "${MFEM_RELEASE_DIR}")
else()
  message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}. Please set it to either 'Debug' or 'Release'.")
endif()
# Include MFEM
include("${MFEM_DIR}/MFEMConfig.cmake")

#--------------------------------------------------------------
# 3. Libraries
# libraries (non-executable)
foreach(lib ${LIB})
  add_library(${lib} STATIC src/${lib}.cpp)
  target_link_libraries(${lib} PRIVATE mfem)
  set(dep_var "LIB_DEP_${lib}")
  if(DEFINED ${dep_var})
    foreach(dep ${${dep_var}})
      target_link_libraries(${lib} PRIVATE ${dep})
    endforeach()
  endif()
  target_link_libraries(core INTERFACE ${lib})
endforeach()
target_link_libraries(core INTERFACE mfem)

#--------------------------------------------------------------
# 4. Executables
foreach(prog ${EXECUTABLES})
  add_executable(${prog} ${prog}.cpp)
  target_link_libraries(${prog} PRIVATE mfem logger)
  set(dep_var "EXE_DEP_${prog}")
  if(DEFINED ${dep_var})
    foreach(dep ${${dep_var}})
      target_link_libraries(${prog} PRIVATE ${dep})
    endforeach()
  endif()
endforeach()
#--------------------------------------------------------------
# 5. Cleanup
# NOT YET IMPLEMENTED
