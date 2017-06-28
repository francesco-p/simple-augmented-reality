# Simple Augmented Reality System

Given a marker in a scene, the program replaces it with the corresponding painting:

![System](./data/system.png "System")

## Demo

In this demo the main operations can be seen:

1. Thresholding (Otsu or binary)
2. Contour detection and filtering
3. Marker recognition (template matching)

![System Demo](./data/cv.gif "System Demo")

## How to run

In order to run it you need OpenCV and CMake installed in your system, then:

```
mkdir build
cd build
cmake ../
make run
```

If you don't have OpenCV installed in the default location (which for me is `/usr/local/lib/`), you can specify the path through: `cmake ../ -DOpenCV_DIR="/path/to/opencv/build"`.


*This is the fourth and last part of the projects that I had to develop for the Computer Vision course.*

