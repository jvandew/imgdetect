classpath = .:../..
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean util

clean:
	-rm *.class

util: BoundingBox.scala PASCALAnnotation.scala PASCALObject.scala PASCALObjectLabel.scala PASCALSize.scala Point.scala Utils.scala
	$(scalac) Utils.scala Point.scala BoundingBox.scala PASCALSize.scala PASCALObjectLabel.scala PASCALObject.scala PASCALAnnotation.scala