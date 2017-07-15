cmake_minimum_required(VERSION 3.0)
project(deep_coder)

# common settings
enable_testing()

# C++ settings
if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
    endif()
else () 
	set(CMAKE_CXX_STANDARD 14)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(src)