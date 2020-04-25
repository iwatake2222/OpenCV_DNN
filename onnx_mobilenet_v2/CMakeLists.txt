cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Compile option
if(NOT MSVC)
	set(CMAKE_C_FLAGS "-Wall")
	set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
	if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
	endif()
endif()

# Create executable file
add_executable(${ProjectName}
	Main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${ProjectName} ${OpenCV_LIBS})
endif()

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/../resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")
