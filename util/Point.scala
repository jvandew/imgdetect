package imgdetect.util

// a point; points can only live in quadrant 1
case class Point (val x: Int, val y: Int) {

  require(x >= 0 && y >= 0, "Error: Points may not contain negative values")

}