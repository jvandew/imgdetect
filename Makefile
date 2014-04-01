classpath = .:share/OpenCV/java/opencv-248.jar
libpath = share/OpenCV/java
copts = -deprecation -optimise -cp $(classpath)
ropts = -cp $(classpath) -Djava.library.path=share/OpenCV/java
scalac = fsc $(copts)

.PHONY: all clean go run test util

all: util test

clean:
	@-$(MAKE) clean -C util -s
	-rm *.class

go: all run

run:
	scala $(ropts) CVTest

test: CVTest.scala
	$(scalac) CVTest.scala

util:
	@-$(MAKE) util -C util --no-print-directory