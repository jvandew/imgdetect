classpath = .:../..:../lib/commons-math3-3.2.jar
scalac += -cp $(classpath) -d ../..

.PHONY: clean util

clean:
	-rm *.class

util: ArrayDist.scala              	  \
			BayesContHOGDetector.scala      \
			BayesDiscHOGDetector.scala      \
			BayesianDetector.scala       	  \
			BoundingBox.scala            	  \
			ContinuousDistribution.scala    \
			ContinuousHOGCell.scala         \
			DependentBayesHOGDetector.scala \
			DirichletHashMapDist.scala   	  \
			DiscreteDistribution.scala   	  \
			DiscreteHOGCell.scala        	  \
			Distribution.scala           	  \
			HashMapDist.scala            	  \
			MultivarNormalDist.scala        \
			PASCALAnnotation.scala       	  \
			PASCALObject.scala           	  \
			PASCALObjectLabel.scala      	  \
			PASCALSize.scala             	  \
			Point.scala                  	  \
			Utils.scala                     \
			VectorType.scala
	$(scalac)                         \
		Utils.scala                     \
		VectorType.scala                \
		ContinuousHOGCell.scala         \
		DiscreteHOGCell.scala           \
		Distribution.scala              \
		ContinuousDistribution.scala    \
		DiscreteDistribution.scala      \
		ArrayDist.scala                 \
		HashMapDist.scala               \
		DirichletHashMapDist.scala      \
		MultivarNormalDist.scala        \
		Point.scala                     \
		BoundingBox.scala               \
		PASCALSize.scala                \
		PASCALObjectLabel.scala         \
		PASCALObject.scala              \
		PASCALAnnotation.scala          \
		Detector.scala                  \
		BayesianDetector.scala          \
		BayesContHOGDetector.scala      \
		BayesDiscHOGDetector.scala      \
		DependentBayesHOGDetector.scala