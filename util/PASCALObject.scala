package imgdetect.util

// Container class for PASCAL annotation objects
case class PASCALObject (val label: PASCALObjectLabel, val center: Point, val box: BoundingBox)