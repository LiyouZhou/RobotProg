cmake_minimum_required(VERSION 3.8)
project(pothole_inspection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
    DIRECTORY
    launch
    params
    rviz
    maps
    models
    waypoints
    DESTINATION share/${PROJECT_NAME}/
)

# Install Python executables
install(
  PROGRAMS
  src/detection_aggregation_node.py
  src/object_detection_node.py
  src/waypoint_mission_node.py
  src/report_generator_node.py
  src/pothole_tracker.py
  src/utils.py
  DESTINATION lib/${PROJECT_NAME}
)

find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection2DArrayWithSourceImage.msg"
  "msg/Pothole.msg"
  "srv/ReportAggregatedDetections.srv"
  "srv/GenerateReport.srv"
  DEPENDENCIES
  vision_msgs
  sensor_msgs
 )

ament_package()
