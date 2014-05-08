package imgdetect.tests

import imgdetect.cvtools.CVTools
import imgdetect.detector.BayesContLocationHOGDetector
import imgdetect.util.{BoundingBox, Negative, PASPerson, Point}
import java.io.{FileInputStream, ObjectInputStream}

object CVTest {

  def displayModel (model: BayesContLocationHOGDetector) : Unit = {

    val cellSize = CVTools.makeSize(40, 40)
    val numBins = 9
    val img = CVTools.newGrayscaleImage(640, 320)(0)

    val posDists = model.distMap(PASPerson)
    val posMeans = posDists.map(_.mean)
    val posHogs = posMeans.map(_.map(_.toFloat)).grouped(8).toArray
    val posimg = CVTools.getModelVisual(img)(posHogs)(cellSize, numBins)(4.0f)(4.0f)

    val negDists = model.distMap(Negative)
    val negMeans = negDists.map(_.mean)
    val negHogs = negMeans.map(_.map(_.toFloat)).grouped(8).toArray
    val negimg = CVTools.getModelVisual(img)(negHogs)(cellSize, numBins)(4.0f)(4.0f)

    CVTools.displayImage(posimg)("positive model")
    CVTools.displayImage(negimg)("negative model")

  }

  def lena () : Unit = {

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

  def main (args: Array[String]) = {

    CVTools.loadLibrary

    args(0) match {
      case "lena" => lena
      case "displayModel" => {

        val in = new ObjectInputStream(new FileInputStream(args(1)))

        in.readObject match {
          case contLocDet: BayesContLocationHOGDetector =>
            displayModel(contLocDet)
          case _ =>
            throw new IllegalArgumentException("A BayesContLocationHOGDetector could not be parsed from " + args(1))
        }

      }
    }
  }

}
