# Copied from https://github.com/abetlen/llama-cpp-python/blob/main/CMakeLists.txt
# Edits made to remove llama.cpp specific calls
cmake_minimum_required(VERSION 3.21)

project(ggml)

function(ggml_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
    )
    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
    )
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

set(BUILD_SHARED_LIBS "On")

set(CMAKE_SKIP_BUILD_RPATH FALSE)

# When building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_SKIP_RPATH FALSE)

# Architecture detection and settings for Apple platforms
if (APPLE)
    # Get the target architecture
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # If CMAKE_OSX_ARCHITECTURES is not set, use the host architecture
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES ${HOST_ARCH} CACHE STRING "Build architecture for macOS" FORCE)
    endif()

    message(STATUS "Host architecture: ${HOST_ARCH}")
    message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

    # Configure based on target architecture
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        # Intel Mac settings
        set(GGML_AVX "OFF" CACHE BOOL "ggml: enable AVX" FORCE)
        set(GGML_AVX2 "OFF" CACHE BOOL "ggml: enable AVX2" FORCE)
        set(GGML_FMA "OFF" CACHE BOOL "ggml: enable FMA" FORCE)
        set(GGML_F16C "OFF" CACHE BOOL "ggml: enable F16C" FORCE)
    endif()

    # Default Metal to OFF
    if(NOT GGML_METAL)
        set(GGML_METAL "OFF" CACHE BOOL "ggml: enable Metal" FORCE)
    endif()

endif()


add_subdirectory(vendor/ggml)

ggml_python_install_target(ggml)

ggml_python_install_target(ggml-base)

ggml_python_install_target(ggml-amx)
ggml_python_install_target(ggml-blas)
ggml_python_install_target(ggml-can)
ggml_python_install_target(ggml-cpu)
ggml_python_install_target(ggml-cuda)
ggml_python_install_target(ggml-hip)
ggml_python_install_target(ggml-kompute)
ggml_python_install_target(ggml-metal)
ggml_python_install_target(ggml-musa)
ggml_python_install_target(ggml-rpc)
ggml_python_install_target(ggml-sycl)
ggml_python_install_target(ggml-vulkan)

# Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
if (WIN32)
    install(
        FILES $<TARGET_RUNTIME_DLLS:ggml>
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib
    )
    install(
        FILES $<TARGET_RUNTIME_DLLS:ggml>
        DESTINATION ${SKBUILD_PLATLIB_DIR}/ggml/lib
    )
endif()
