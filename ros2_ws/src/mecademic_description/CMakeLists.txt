cmake_minimum_required(VERSION 3.10)
project(mecademic_description)

find_package(ament_cmake REQUIRED)

# Install shareable assets
install(DIRECTORY
  urdf
  meshes
  DESTINATION share/${PROJECT_NAME}
)

# Register in ament index (recommended)
install(FILES package.xml
        DESTINATION share/${PROJECT_NAME})
install(FILES resource/${PROJECT_NAME}
        DESTINATION share/ament_index/resource_index/packages)

ament_package()
