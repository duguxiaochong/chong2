# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/share/cmake-3.15/bin/cmake

# The command to remove a file.
RM = /usr/share/cmake-3.15/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chongge/myWolfTeamWork-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chongge/myWolfTeamWork-main/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov5.dir/flags.make

CMakeFiles/yolov5.dir/calibrator.cpp.o: CMakeFiles/yolov5.dir/flags.make
CMakeFiles/yolov5.dir/calibrator.cpp.o: ../calibrator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chongge/myWolfTeamWork-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov5.dir/calibrator.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5.dir/calibrator.cpp.o -c /home/chongge/myWolfTeamWork-main/calibrator.cpp

CMakeFiles/yolov5.dir/calibrator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5.dir/calibrator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chongge/myWolfTeamWork-main/calibrator.cpp > CMakeFiles/yolov5.dir/calibrator.cpp.i

CMakeFiles/yolov5.dir/calibrator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5.dir/calibrator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chongge/myWolfTeamWork-main/calibrator.cpp -o CMakeFiles/yolov5.dir/calibrator.cpp.s

# Object files for target yolov5
yolov5_OBJECTS = \
"CMakeFiles/yolov5.dir/calibrator.cpp.o"

# External object files for target yolov5
yolov5_EXTERNAL_OBJECTS =

libyolov5.a: CMakeFiles/yolov5.dir/calibrator.cpp.o
libyolov5.a: CMakeFiles/yolov5.dir/build.make
libyolov5.a: CMakeFiles/yolov5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chongge/myWolfTeamWork-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libyolov5.a"
	$(CMAKE_COMMAND) -P CMakeFiles/yolov5.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov5.dir/build: libyolov5.a

.PHONY : CMakeFiles/yolov5.dir/build

CMakeFiles/yolov5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov5.dir/clean

CMakeFiles/yolov5.dir/depend:
	cd /home/chongge/myWolfTeamWork-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chongge/myWolfTeamWork-main /home/chongge/myWolfTeamWork-main /home/chongge/myWolfTeamWork-main/build /home/chongge/myWolfTeamWork-main/build /home/chongge/myWolfTeamWork-main/build/CMakeFiles/yolov5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov5.dir/depend

