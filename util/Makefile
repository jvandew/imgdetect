classpath = .:../..
copts = -deprecation -optimise -cp $(classpath) -d ../..
scalac = fsc $(copts)

.PHONY: clean util

clean:
	-rm *.class

util: PASCALAnnotation.scala PASCALObject.scala PASCALSize.scala
	$(scalac) PASCALObject.scala PASCALSize.scala PASCALAnnotation.scala