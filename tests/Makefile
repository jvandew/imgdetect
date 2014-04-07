# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../share/OpenCV/java/opencv-248.jar
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean tests

clean:
	-rm *.class

tests: CVTest.scala
	$(scalac) CVTest.scala