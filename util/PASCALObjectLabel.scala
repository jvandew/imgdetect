package imgdetect.util

// ADT for PASCAL object types (granted right now we only have one type but someday...)
sealed abstract class PASCALObjectLabel {

  override def toString: String = this.getClass.getSimpleName

}

case object Negative extends PASCALObjectLabel

// hierarchical structure not represented to allow for easy hashing of labels
case object PASPerson extends PASCALObjectLabel
case object UprightPerson extends PASCALObjectLabel
case object BikingPerson extends PASCALObjectLabel