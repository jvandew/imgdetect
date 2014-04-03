classpath = .:share/OpenCV/java/opencv-248.jar
libpath = share/OpenCV/java
copts = -deprecation -optimise -cp $(classpath)
ropts = -cp $(classpath) -Djava.library.path=share/OpenCV/java
scalac = fsc $(copts)

.PHONY: all clean go run test train util

all: util train test

clean:
	@-$(MAKE) clean -C util -s
	@-$(MAKE) clean -C train -s
	-rm *.class

go: all run

run:
	scala $(ropts) CVTest

test: CVTest.scala
	$(scalac) CVTest.scala

train: util
	@-$(MAKE) train -C train --no-print-directory

util:
	@-$(MAKE) util -C util --no-print-directory