import scala.util.matching.Regex.Match

// companion class for PASCALAnnotation
object PASCALAnnotation {

  /* A regex for parsing the PASCAL Annotation file format
   * This regex parses the following groups:
   * 1: image filename
   * 2: size x-value
   * 3: size y-value
   * 4: size c-value
   * 5: database
   * 6: number of objects
   * 7: object label array
   * 8: individual object details
   */
  val annotationRegex = ("""# PASCAL Annotation Version 1.00\n""" +
                         """\n""" +
                         """Image filename : "([\d\w/]+)"\n""" +
                         """Image size \(X x Y x C\) : (\d+) x (\d+) x (\d+)\n""" +
                         """Database : "([\s\w]+)"\n""" +
                         """Objects with ground truth : (\d+) ({(?: *"[\d\w]+" *,)* *"[\d\w]+" *})\n""" +
                         """\n""" +
                         """# Note that there might be other objects in the image\n""" +
                         """# for which ground truth data has not been provided.\n""" +
                         """\n""" +
                         """# Top left pixel co-ordinates : (0, 0)\n""" +
                         """([.\n]*)""").r

  // A regex for matching a label array element
  val labelArrayElementRegex = """ *"([\d\w]+)" *""".r

  /* A regex for parsing individual PASCAL Annotation objects
   * This regex parses the following groups:
   * 1: object number
   * 2: object label
   * 3: original (more specific) object label
   * 4: center x-value
   * 5: center y-value
   * 6: box top left corner x-value
   * 7: box top left corner y-value
   * 8: box bottom right corner x-value
   * 9: box bottom right corner y-value
   */
  val objectRegex = ("""\n""" +
                     """# Details for object \d+ \("[\d\w]+"\)\n""" +
                     """# Center point -- not available in other PASCAL databases -- refers\n""" +
                     """# to person head center\n""" +
                     """Original label for object (\d+) "([\d\w]+)" : "([\d\w]+)"\n""" +
                     """Center point on object \1 "\2" \(X, Y\) : \((\d+), (\d+)\)\n""" +
                     """Bounding box for object \1 "\2" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)\n""").r

  def apply (image: String, size: PASCALSize, objects: List[PASCALObject]) : PASCALAnnotation =
    new PASCALAnnotation(image, size, objects)

  def parseFile (filePath: String) : PASCALAnnotation = {
    val contents = Utils.readFile(filePath)
    val annotationRegex(filename, sizexStr, sizeyStr, sizecStr,
                        database, numObjsStr, labelArray, objDetails) = contents

    val pascalLabels = labelArrayElementRegex.findAllMatchIn(labelArray)
    val matches = objectRegex.findAllMatchIn(objDetails)

    // sanity check
    if (matches.length != pascalLabels.length || matches.length != numObjsStr.toInt) {
      throw new IllegalArgumentException("Wrong number of labels for annotation objects")
    }

    val size = PASCALSize(sizexStr.toInt, sizeyStr.toInt, sizecStr.toInt)

    val pascalObjs = pascalLabels.zip[Match](matches).map { labelStr_mat =>

      val List(objNumStr, labelStr_mat._1, specLabelStr, centerxStr, centeryStr,
               topLeftxStr, topLeftyStr, botRightxStr, botRightyStr) = labelStr_mat._2.subgroups

      val label = specLabelStr match {
        case "PASPerson" => PASPerson
        case "UprightPerson" => UprightPerson
        case other =>
          throw new IllegalArgumentException("Unrecognized object label: " + other)
      }

      val center = Point(centerxStr.toInt, centeryStr.toInt)
      val topLeft = Point(topLeftxStr.toInt, topLeftyStr.toInt)
      val botRight = Point(botRightxStr.toInt, botRightyStr.toInt)
      val box = BoundingBox(topLeft, botRight)

      PASCALObject(label, center, box)
    }

    PASCALAnnotation(filename, size, pascalObjs.toList)
  }

}

// a container object for PASCAL image annotation
class PASCALAnnotation (val image: String, val size: PASCALSize, val objects: List[PASCALObject]) {

  def this (anno: PASCALAnnotation) = this(anno.image, anno.size, anno.objects)

  def this (annotationFile: String) = this(PASCALAnnotation.parseFile(annotationFile))

}