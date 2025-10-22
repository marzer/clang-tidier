cmake_minimum_required(VERSION 3.16)
project(clang_tidier_test_project LANGUAGES CXX C)

add_executable(test_exe main.cpp)
set_target_properties(test_exe PROPERTIES EXPORT_COMPILE_COMMANDS ON)
