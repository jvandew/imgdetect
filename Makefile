classpath = .:share/OpenCV/java/opencv-248.jar
libpath = share/OpenCV/java
copts = -deprecation -optimise -cp $(classpath)
ropts = -cp $(classpath) -Djava.library.path=share/OpenCV/java
scalac = fsc $(copts)

.PHONY: all clean go run test

all: test

clean:
	-rm *.class

go: all run

run:
	scala $(ropts) CVTest

test: CVTest.scala
	$(scalac) CVTest.scala
