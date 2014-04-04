classpath = .:../..:../share/OpenCV/java/opencv-248.jar
libpath = share/OpenCV/java
copts = -deprecation -optimise -cp $(classpath) -d ../..
ropts = -cp $(classpath) -Djava.library.path=$(libpath)
scalac = fsc $(copts)

.PHONY: clean run cvtest

clean:
	-rm *.class

run:
	scala $(ropts) CVTest

cvtest: CVTest.scala
	$(scalac) CVTest.scala