classpath = .:../..:../lib/commons-math3-3.2.jar
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean util

clean:
	-rm *.class

util: ArrayDist.scala              \
			BayesHOGDetector.scala       \
			BayesianDetector.scala       \
			BoundingBox.scala            \
			DirichletHashMapDist.scala   \
			DiscreteDistribution.scala   \
			DiscreteHOGCell.scala        \
			Distribution.scala           \
			HashMapDist.scala            \
			PASCALAnnotation.scala       \
			PASCALObject.scala           \
			PASCALObjectLabel.scala      \
			PASCALSize.scala             \
			Point.scala                  \
			Utils.scala
	$(scalac)                        \
		Utils.scala                    \
		DiscreteHOGCell.scala          \
		Distribution.scala             \
		DiscreteDistribution.scala     \
		ArrayDist.scala                \
		HashMapDist.scala              \
		DirichletHashMapDist.scala     \
		Point.scala                    \
		BoundingBox.scala              \
		PASCALSize.scala               \
		PASCALObjectLabel.scala        \
		PASCALObject.scala             \
		PASCALAnnotation.scala         \
		Detector.scala                 \
		BayesianDetector.scala         \
		BayesHOGDetector.scala