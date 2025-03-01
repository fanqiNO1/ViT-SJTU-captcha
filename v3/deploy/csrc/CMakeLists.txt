cmake_minimum_required(VERSION 3.12)

project(ViT-SJTU-captcha-v3-deploy
        DESCRIPTION "Deploy ViT-SJTU-captcha-v3 using C++"
        HOMEPAGE_URL "https://github.com/fanqiNO1/ViT-SJTU-captcha/tree/main/v3")

# Set options
option(BUILD_SEGMENT "Build segment library" ON)
option(BUILD_TEST "Build test" ON)

# Find packages
find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS})

# Set C++ standard
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Add subdirectories
add_subdirectory(ggml)

# Build library
# add_library(vit_captcha SHARED vit_captcha.cpp)
# target_include_directories(vit_captcha PUBLIC .)
# target_link_libraries(vit_captcha PUBLIC ggml)
# target_link_libraries(vit_captcha PUBLIC ${OpenCV_LIBS})
# target_compile_features(vit_captcha PUBLIC cxx_std_17)

# Build library
if (BUILD_SEGMENT)
    add_library(segment SHARED segment.cpp)
    target_include_directories(segment PUBLIC .)
    target_compile_features(segment PUBLIC cxx_std_17)
endif()

# Install
if (BUILD_SEGMENT)
    install(TARGETS segment
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

# Build test
if (BUILD_TEST)
    add_executable(segment_test tests/segment_test.cpp segment.cpp)
    target_include_directories(segment_test PUBLIC .)
    target_compile_features(segment_test PUBLIC cxx_std_17)
endif()
