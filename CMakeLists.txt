cmake_minimum_required(VERSION 3.18)

project(fluid_sandbox
    LANGUAGES CXX CUDA
)

# --------------------------------------------------
# Language standards
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --------------------------------------------------
# MSVC runtime (important for static builds)
# --------------------------------------------------
if (MSVC)
    # Use /MT and /MTd instead of /MD
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# --------------------------------------------------
# Find OpenGL
# --------------------------------------------------
find_package(OpenGL REQUIRED)

# --------------------------------------------------
# GLFW: build static
# --------------------------------------------------
include(FetchContent)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

# ?? This makes it STATIC
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.9
)

FetchContent_MakeAvailable(glfw)


# --------------------------------------------------
# CUDA Toolkit
# --------------------------------------------------
find_package(CUDAToolkit REQUIRED)

#add_compile_definitions(DIRECTINPUT_VERSION=0x0800)
#add_compile_definitions(__CUDACC__)

# --------------------------------------------------
# Sources (VS filters)
# --------------------------------------------------
file(GLOB_RECURSE fluid_H include/*.h include/*.cuh) 
file(GLOB_RECURSE fluid_SRC src/*.cpp src/*.c src/*.cu)

source_group("includes" FILES ${fluid_H})
source_group("src" FILES ${fluid_SRC})

set(GLAD_SRC
    glad/include/glad/glad.h
    glad/include/KHR/khrplatform.h
    glad/src/glad.c
)

source_group("GLAD" FILES ${GLAD_SRC})

set(config_file config.cfg)

# --------------------------------------------------
# Executable
# --------------------------------------------------
add_executable(${PROJECT_NAME} ${fluid_H} ${fluid_SRC} ${GLAD_SRC})

# --------------------------------------------------
# Config file
# --------------------------------------------------
configure_file(config.cfg . COPYONLY)

target_sources(${PROJECT_NAME} PRIVATE ${config_file})

# --------------------------------------------------
# Link libraries
# --------------------------------------------------
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        glfw              # static GLFW
        OpenGL::GL
        CUDA::cudart
)

target_include_directories(${PROJECT_NAME} PRIVATE include glad/include)

# --------------------------------------------------
# CUDA settings
# --------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# --------------------------------------------------
# Platform-specific system libs (required for static GLFW)
# --------------------------------------------------
if (WIN32)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            user32
            gdi32
            shell32
    )
elseif (UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            dl
            X11
    )
endif()


# --------------------------------------------------
# Visual Studio startup project
# --------------------------------------------------
if (MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}
    )
endif()
