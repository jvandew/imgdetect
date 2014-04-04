package imgdetect.cvtest

import imgdetect.cvtools.CVTools

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

    CVTools.computeAndDisplayHOG(img)(winSize, blockSize, blockStride, cellSize, numBins)(1.2f)("bs = 32x32")
  }

}
