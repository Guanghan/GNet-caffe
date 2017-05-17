CXX = g++
CXXFLAGS = -Wall -Werror
OPENCVLIBS = -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/
DEPENDENCIES = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching
DEPS = utils.h

default: build

build:
	$(CXX) -c test_HOG.cpp
	$(CXX) -c utils.cpp
	$(CXX) $(CXXFLAGS) $(OPENCVLIBS) -g test_HOG.o utils.o $(DEPENDENCIES) -o demo

run: build
	./test_HOG

clean:
	@- $(RM) -f test_HOG
