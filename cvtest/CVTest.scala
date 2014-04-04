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
    var winStride = CVTools.makeSize(0, 0)
    var blockSize = CVTools.makeSize(32, 32)
    var blockStride = CVTools.makeSize(32, 32)
    var cellSize = CVTools.makeSize(32, 32)
    var numBins = 9

    CVTools.computeAndDisplayHOGFull(img)(winSize, winStride, blockSize, blockStride, cellSize, numBins)(1.2f)("full image")

    val box = BoundingBox(Point(3, 4), Point(15, 10))

    CVTools.computeAndDisplayHOGBox(img)(box)(winSize, winStride, blockSize, blockStride, cellSize, numBins)(1.2f)("full image bbox")

    winSize = CVTools.makeSize(256, 256)
    winStride = CVTools.makeSize(256, 256)

    CVTools.computeAndDisplayHOGFull(img)(winSize, winStride, blockSize, blockStride, cellSize, numBins)(1.2f)("4 windows")

  }

}
