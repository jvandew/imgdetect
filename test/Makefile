# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../share/OpenCV/java/opencv-249.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean test

clean:
	-rm *.class

test: TestBayesSuper.scala
	$(scalac) TestBayesSuper.scala