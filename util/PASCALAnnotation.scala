// companion class for PASCALAnnotation
object PASCALAnnotation {

  def apply (image: String) (size: PASCALSize) (objects: Array[PASCALObject]) : PASCALAnnotation =
    new PASCALAnnotation(image, size, objects)

  def parseFile (filePath: String) : PASCALAnnotation = {
    throw new Exception("NYI")
  }

}

// a container object for PASCAL image annotation
class PASCALAnnotation (val image: String, val size: PASCALSize, val objects: Array[PASCALObject]) {

  def this (anno: PASCALAnnotation) = this(anno.image, anno.size, anno.objects)

  def this (annotationFile: String) = this(PASCALAnnotation.parseFile(annotationFile))

}