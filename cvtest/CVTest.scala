package imgdetect.cvtest

import imgdetect.cvtools.CVTools
import imgdetect.util.{BoundingBox, Point}

object CVTest {

  def main (args: Array[String]) = {

    CVTools.loadLibrary

    // TODO(jacob) figure out how to get this path dynamically
    // println(ClassLoader.getSystemResource("lena.png").getPath)   // crashes w/ NPE
    val img = CVTools.imreadGrayscale("resources/lena.png")

    var winSize = CVTools.makeSize(512, 512) // size of lena
    var blockSize = CVTools.makeSize(32, 32)
    var blockStride = CVTools.makeSize(32, 32)
    var cellSize = CVTools.makeSize(32, 32)
    var numBins = 9

    CVTools.computeAndDisplayHOGFull(img)(winSize, blockSize, blockStride, cellSize, numBins)(1.2f)("bs = 32x32")

    val box = BoundingBox(Point(3, 4), Point(15, 10))

    CVTools.computeAndDisplayHOGBox(img)(box)(winSize, blockSize, blockStride, cellSize, numBins)(1.2f)("bb: bs = 32x32")
  }

}
