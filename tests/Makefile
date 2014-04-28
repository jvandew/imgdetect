# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../share/OpenCV/java/opencv-248.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean tests

clean:
	-rm *.class

tests: CVTest.scala
	$(scalac) CVTest.scala