cmake_minimum_required(VERSION 3.8)
project(robot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Optional: WiringPi
find_library(WIRINGPI_LIB wiringPi)

# ───── Executable ─────
add_executable(servo_node src/servo.cpp)

add_executable(motor_controller_node src/motor_controller.cpp)

# Link libserial libraries
target_link_libraries(motor_controller_node PRIVATE ${WIRINGPI_LIB})

# Must use keyword form now
target_link_libraries(servo_node PRIVATE ${WIRINGPI_LIB})

# ───── Installation ─────
install(TARGETS
  servo_node
  motor_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
