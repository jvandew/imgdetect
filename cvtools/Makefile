classpath = .:../..:../share/OpenCV/java/opencv-248.jar
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean cvtools

clean:
	-rm *.class

cvtools: CVTools.scala
	$(scalac) CVTools.scala