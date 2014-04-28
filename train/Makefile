# unfortunately Scala mandates our classpath still include the jarchive
classpath = .:../..:../share/OpenCV/java/opencv-248.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean train

clean:
	-rm *.class

train: TrainBayesSuper.scala
	$(scalac) TrainBayesSuper.scala