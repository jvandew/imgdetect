# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../lib/commons-math3-3.2.jar:../share/OpenCV/java/opencv-249.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean train

clean:
	-rm *.class

train: TrainBayesSuper.scala
	$(scalac) TrainBayesSuper.scala