package imgdetect.util

// ADT for PASCAL object types (granted right now we only have one type but someday...)
sealed trait PASCALObjectLabel

case object Negative extends PASCALObjectLabel

// hierarchical structure not represented to allow for easy hashing of labels
case object PASPerson extends PASCALObjectLabel
case object UprightPerson extends PASCALObjectLabel
case object BikingPerson extends PASCALObjectLabel