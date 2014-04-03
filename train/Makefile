classpath = .:../..
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean train

clean:
	-rm *.class

train: TrainBayesSuper.scala
	$(scalac) TrainBayesSuper.scala