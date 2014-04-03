package imgdetect.util

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
                         """Image filename : "(.+?)"\n""" +
                         """Image size \(X x Y x C\) : (\d+) x (\d+) x (\d+)\n""" +
                         """Database : "(.+?)"\n""" +
                         """Objects with ground truth : (\d+) (\{(?: *"\w+" *)*\})\n""" +
                         """\n""" +
                         """# Note that there might be other objects in the image\n""" +
                         """# for which ground truth data has not been provided.\n""" +
                         """\n""" +
                         """# Top left pixel co-ordinates : \(0, 0\)\n""" +
                         """([\S\s]*)""").r

  // A regex for matching a label array element
  val labelArrayElementRegex = """\w+""".r

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
                     """# Details for object \d+ \("\w+"\)\n""" +
                     """# Center point -- not available in other PASCAL databases -- refers\n""" +
                     """# to person head center\n""" +
                     """Original label for object (\d+) "(\w+)" : "(\w+)"\n""" +
                     """Center point on object \d+ "\w+" \(X, Y\) : \((\d+), (\d+)\)\n""" +
                     """Bounding box for object \d+ "\w+" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)\n""").r


  def apply (image: String, size: PASCALSize, objects: List[PASCALObject]) : PASCALAnnotation =
    new PASCALAnnotation(image, size, objects)


  // inefficient - can do without allocating a separate string, array, and two lists
  // but we don't really care for a one time operation, this is cleaner
  def parseAnnotationList(inriaHome: String, listFile: String) : List[PASCALAnnotation] = {

    val path = Utils.joinPath(inriaHome, listFile)
    val filenames = Utils.readFile(path).split('\n')
    filenames.toList.map(parseFile(inriaHome, _))

  }


  def parseFile (inriaHome: String, filePath: String) : PASCALAnnotation = {

    val path = Utils.joinPath(inriaHome, filePath)
    val contents = Utils.readFile(path)
    val annotationRegex(filename, sizexStr, sizeyStr, sizecStr,
                        database, numObjsStr, labelArray, objDetails) = contents

    val pascalLabels = labelArrayElementRegex.findAllMatchIn(labelArray).toList
    val matches = objectRegex.findAllMatchIn(objDetails).toList

    // sanity check
    if (matches.length != pascalLabels.length || matches.length != numObjsStr.toInt) {
      throw new IllegalArgumentException("Wrong number of labels for annotation objects")
    }

    val size = PASCALSize(sizexStr.toInt, sizeyStr.toInt, sizecStr.toInt)

    val pascalObjs = pascalLabels.zip(matches).map { labelStr_mat =>

      // val List(objNumStr, labelStr_mat._1, specLabelStr, centerxStr, centeryStr,
      //          topLeftxStr, topLeftyStr, botRightxStr, botRightyStr) = labelStr_mat._2.subgroups
      val fields = labelStr_mat._2.subgroups
      val objNumStr = fields(0)
      val specLabelStr = fields(2)
      val centerxStr = fields(3)
      val centeryStr = fields(4)
      val topLeftxStr = fields(5)
      val topLeftyStr = fields(6)
      val botRightxStr = fields(7)
      val botRightyStr = fields(8)

      val label = specLabelStr match {
        case "PASperson" => PASPerson
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

  def this (inriaHome: String, annoFile: String) = this(PASCALAnnotation.parseFile(inriaHome, annoFile))

}