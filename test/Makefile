# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../share/OpenCV/java/opencv-248.jar
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean test

clean:
	-rm *.class

test: TestBayesSuper.scala
	$(scalac) TestBayesSuper.scala