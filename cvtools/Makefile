classpath = .:../..:../share/OpenCV/java/opencv-248.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean cvtools

clean:
	-rm *.class

cvtools: CVTools.scala
	$(scalac) CVTools.scala