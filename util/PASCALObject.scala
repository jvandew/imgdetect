// ADT for PASCAL object types (granted right now we only have one type but someday...)
sealed trait PASCALObject

sealed trait PASPerson extends PASCALObject

case object UprightPerson extends PASPerson