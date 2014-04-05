package imgdetect.train

import imgdetect.cvtools.CVTools
import imgdetect.util.{BoundingBox, DiscreteHOGCell, HashMapDist, PASCALAnnotation, Point, Utils}
import java.io.File
import org.opencv.core.{Mat, Size}

// Train a supervised Bayesian detector
object TrainBayesSuper {

  /* Train a detector. Arguments:
   * 0: -norm = train on normalized dataset or -full = train on original full images
   * 1: INRIA dataset location
   * 2: number of bins to use in HOG descriptor
   * 3: number of partitions to use for each gradient bin. this is used to take
   *    the space of HOG cell descriptors from a "continuous" to a discrete space
   *
   *    Note the total number of discetized HOG descriptors will be
   *    [num partitions]^[num gradient bins]
   */
  def main (args: Array[String]) : Unit = {

    CVTools.loadLibrary

    val inriaHome = args(1)
    val numBins = args(2).toInt
    val numParts = args(3).toInt

    val winSize = CVTools.makeSize(64, 128)
    val winStride = CVTools.makeSize(0, 0)
    val blockSize = CVTools.makeSize(16, 16)
    val blockStride = CVTools.makeSize(16, 16)
    val cellSize = CVTools.makeSize(8, 8)

    args(0) match {
      case "-norm" => {

        val posFiles = new File(inriaHome, "train_64x128_H96/pos")
        val posImages = posFiles.listFiles
        val dist = new HashMapDist[DiscreteHOGCell]

        var counter = 0

        posImages.foreach { imgFile =>

          counter += 1
          val path = imgFile.getPath

          println("handling image " + counter + ": " + path)

          val img = CVTools.imreadGreyscale(path)

          // normalized images are padded by 16 pixels on each side
          val cropBox = BoundingBox(Point(16, 16), 64, 128)
          val cropped = CVTools.cropImage(img, cropBox)

          // compute HOG descriptors
          val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)
          val discreteHOGs = descs.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts))

          dist.addWords(discreteHOGs)

        }

        println("\nComputed " + dist.totalUnique + " unique descriptors:\n")
        dist.display

      }

      case "-full" => {

        val annotations = PASCALAnnotation.parseAnnotationList(inriaHome, "Train/annotations.lst")

        println("parse successful?")

      }

      case other =>
        throw new IllegalArgumentException("Unrecognized parameter: " + other)
    }

  }

}