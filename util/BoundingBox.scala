package imgdetect.util

// companion object for BoundingBox
object BoundingBox {

  def apply (topLeft: Point, bottomRight: Point) : BoundingBox =
    new BoundingBox(topLeft, bottomRight)

  def apply (topLeft: Point, width: Int, height: Int) : BoundingBox =
    new BoundingBox(topLeft, width, height)

}

// a bounding box is defined by two points or a point and dimensions
class BoundingBox (val topLeft: Point, val bottomRight: Point) {

  require(topLeft.x <= bottomRight.x && topLeft.y <= bottomRight.y,
          "Error: This box is inverted and has negative dimensions")

  val width = bottomRight.x - topLeft.x
  val height = bottomRight.y - topLeft.y

  def this (topLeft: Point, width: Int, height: Int) =
    this(topLeft, Point(topLeft.x + width, topLeft.y + height))

  def this (box: BoundingBox, scale: Float) =
    this (box.topLeft, (box.width * scale).toInt, (box.height * scale).toInt)

}