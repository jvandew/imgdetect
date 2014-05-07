classpath = .:../..
scalac += -cp $(classpath) -d ../..

.PHONY: clean util

clean:
	-rm *.class

util: BoundingBox.scala        \
			ContinuousHOGCell.scala  \
			DiscreteHOGCell.scala    \
			PASCALAnnotation.scala   \
			PASCALObject.scala       \
			PASCALObjectLabel.scala  \
			PASCALSize.scala         \
			Point.scala              \
			Utils.scala              \
			VectorType.scala
	$(scalac)                  \
		Utils.scala              \
		VectorType.scala         \
		ContinuousHOGCell.scala  \
		DiscreteHOGCell.scala    \
		Point.scala              \
		BoundingBox.scala        \
		PASCALSize.scala         \
		PASCALObjectLabel.scala  \
		PASCALObject.scala       \
		PASCALAnnotation.scala