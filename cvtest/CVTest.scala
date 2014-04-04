package imgdetect.cvtest

import imgdetect.cvtools.CVTools
import org.opencv.core.Size
import org.opencv.highgui.Highgui

object CVTest {

  def main (args: Array[String]) = {

    CVTools.loadLibrary

    // TODO(jacob) figure out how to get this path dynamically
    // println(ClassLoader.getSystemResource("lena.png").getPath)   // crashes w/ NPE
    val img = Highgui.imread("resources/lena.png", Highgui.CV_LOAD_IMAGE_GRAYSCALE)

    var winSize = new Size(512, 512) // size of lena
    var blockSize = new Size(32, 32)
    var blockStride = new Size(32, 32)
    var cellSize = new Size(32, 32)
    var numBins = 9

    CVTools.computeAndDisplayHOG(img)(winSize, blockSize, blockStride, cellSize, numBins)(1.2f)("bs = 32x32")
  }

}
