cmake_minimum_required(VERSION 3.16)

project(qwen35_dflash_acl_runner VERSION 1.0.0 LANGUAGES CXX)

option(QWEN35_DFLASH_BUILD_ACL_RUNNER "Build the AscendCL executable" ON)
option(QWEN35_DFLASH_BUILD_TESTS "Build host-only scheduler tests" ON)
set(ASCENDCL_ROOT "" CACHE PATH "Root of the active CANN/AscendCL toolkit")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(qwen35_dflash_core STATIC
  src/generation.cpp
  src/sha256.cpp
)
target_include_directories(qwen35_dflash_core PUBLIC include)
target_compile_options(qwen35_dflash_core PRIVATE -Wall -Wextra -Wpedantic)

if(QWEN35_DFLASH_BUILD_ACL_RUNNER)
  set(_ascendcl_hints "${ASCENDCL_ROOT}")
  foreach(_variable ASCEND_HOME_PATH CANN_HOME ASCEND_TOOLKIT_HOME)
    if(DEFINED ENV{${_variable}})
      list(APPEND _ascendcl_hints "$ENV{${_variable}}")
    endif()
  endforeach()
  list(REMOVE_ITEM _ascendcl_hints "")

  find_path(ASCENDCL_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS ${_ascendcl_hints}
    PATH_SUFFIXES include
  )
  find_library(ASCENDCL_LIBRARY
    NAMES ascendcl
    HINTS ${_ascendcl_hints}
    PATH_SUFFIXES lib64 lib
  )
  if(NOT ASCENDCL_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY)
    message(FATAL_ERROR
      "AscendCL headers/library not found. Activate the declared CANN profile "
      "or pass -DASCENDCL_ROOT=<toolkit-root>. For host tests only, pass "
      "-DQWEN35_DFLASH_BUILD_ACL_RUNNER=OFF."
    )
  endif()

  add_executable(qwen35_dflash_acl_runner
    src/acl_executor.cpp
    src/main.cpp
  )
  target_include_directories(qwen35_dflash_acl_runner PRIVATE
    "${ASCENDCL_INCLUDE_DIR}"
  )
  target_link_libraries(qwen35_dflash_acl_runner PRIVATE
    qwen35_dflash_core
    "${ASCENDCL_LIBRARY}"
  )
  target_compile_definitions(qwen35_dflash_acl_runner PRIVATE
    QWEN35_DFLASH_RUNNER_VERSION="${PROJECT_VERSION}"
  )
  target_compile_options(qwen35_dflash_acl_runner PRIVATE
    -Wall -Wextra -Wpedantic
  )
endif()

if(QWEN35_DFLASH_BUILD_TESTS)
  enable_testing()
  add_executable(qwen35_dflash_cpp_tests tests/test_generation.cpp)
  target_link_libraries(qwen35_dflash_cpp_tests PRIVATE qwen35_dflash_core)
  target_compile_options(qwen35_dflash_cpp_tests PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME qwen35_dflash_cpp_tests COMMAND qwen35_dflash_cpp_tests)

  add_library(qwen35_dflash_fake_acl STATIC tests/fake_acl/fake_acl.cpp)
  target_include_directories(qwen35_dflash_fake_acl PUBLIC tests/fake_acl)
  target_compile_options(qwen35_dflash_fake_acl PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(qwen35_dflash_acl_runner_fake
    src/acl_executor.cpp
    src/main.cpp
  )
  target_include_directories(qwen35_dflash_acl_runner_fake PRIVATE tests/fake_acl)
  target_link_libraries(qwen35_dflash_acl_runner_fake PRIVATE
    qwen35_dflash_core
    qwen35_dflash_fake_acl
  )
  target_compile_definitions(qwen35_dflash_acl_runner_fake PRIVATE
    QWEN35_DFLASH_RUNNER_VERSION="${PROJECT_VERSION}-fake-acl"
  )
  target_compile_options(qwen35_dflash_acl_runner_fake PRIVATE
    -Wall -Wextra -Wpedantic
  )

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fake.om" "fake-om")
  file(SHA256 "${CMAKE_CURRENT_BINARY_DIR}/fake.om" _fake_om_sha256)
  add_test(
    NAME qwen35_dflash_fake_acl_runner
    COMMAND "${CMAKE_COMMAND}"
      -DRUNNER=$<TARGET_FILE:qwen35_dflash_acl_runner_fake>
      -DMODEL=${CMAKE_CURRENT_BINARY_DIR}/fake.om
      -DMODEL_SHA=${_fake_om_sha256}
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/fake-runner-report.json
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fake_runner.cmake
  )
endif()
