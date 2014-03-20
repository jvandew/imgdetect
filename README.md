imgdetect
=========

Object recognition in images. Currently just a work in progress Bayesian topic model detector built for a class project.

The included test file can compiled and run using:

    scalac -cp .:share/OpenCV/java/opencv-248.jar CVTest.scala
    scala -cp .:share/OpenCV/java/opencv-248.jar -Djava.library.path=share/OpenCV/java CVTest

*Note:*  
This project uses [OpenCV](http://opencv.org/) for some image processing. The included libraries were compiled for 64-bit Mac OS X. Your mileage may vary.
