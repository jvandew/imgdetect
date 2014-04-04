classpath = .:../..:../share/OpenCV/java/opencv-248.jar
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean cvtest

clean:
	-rm *.class

cvtest: CVTest.scala
	$(scalac) CVTest.scala