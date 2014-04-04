package imgdetect.util

// a point; points can only live in quadrant 1
case class Point (val x: Int, val y: Int) {

  if (x < 0 || y < 0) {
    throw new IllegalArgumentException("Error: Points may not contain negative values")
  }

}