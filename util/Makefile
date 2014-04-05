classpath = .:../..
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean util

clean:
	-rm *.class

util: ArrayDist.scala BoundingBox.scala DiscreteDistribution.scala DiscreteHOGCell.scala  \
			HashMapDist.scala PASCALAnnotation.scala PASCALObject.scala PASCALObjectLabel.scala \
			PASCALSize.scala Point.scala Utils.scala
	$(scalac) \
		Utils.scala DiscreteHOGCell.scala DiscreteDistribution.scala ArrayDist.scala \
		HashMapDist.scala Point.scala BoundingBox.scala PASCALSize.scala             \
		PASCALObjectLabel.scala PASCALObject.scala PASCALAnnotation.scala