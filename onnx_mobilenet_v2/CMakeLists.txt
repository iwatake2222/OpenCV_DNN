cmake_minimum_required(VERSION 3.0)

if (WIN32 )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Create executable file
set(PROJECT_NAME main)
project(${PROJECT_NAME})
add_executable(${PROJECT_NAME}
	Main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/resource/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/resource/")
