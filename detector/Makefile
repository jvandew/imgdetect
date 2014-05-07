classpath = .:../..
scalac += -cp $(classpath) -d ../..

.PHONY: clean detector

clean:
	-rm *.class

detector: BayesContHOGDetector.scala          \
					BayesContLocationHOGDetector.scala  \
					BayesDiscHOGDetector.scala          \
					BayesianDetector.scala       	      \
					Detector.scala
	$(scalac)                             \
		Detector.scala                      \
		BayesianDetector.scala              \
		BayesContHOGDetector.scala          \
		BayesContLocationHOGDetector.scala  \
		BayesDiscHOGDetector.scala