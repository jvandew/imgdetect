// ADT for PASCAL object types (granted right now we only have one type but someday...)
sealed trait PASCALObjectLabel

sealed trait PASPerson extends PASCALObjectLabel

case object UprightPerson extends PASPerson