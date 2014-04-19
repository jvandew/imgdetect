package imgdetect.train

import imgdetect.cvtools.CVTools
import imgdetect.util.{BayesHOGDetector, BoundingBox, DirichletHashMapDist, DiscreteHOGCell,
                       HashMapDist, Negative, PASCALAnnotation, PASCALObjectLabel, PASPerson,
                       Point, Utils}
import java.io.{File, FileOutputStream, ObjectOutputStream}
import org.opencv.core.{Mat, Size}
import scala.collection.parallel.ThreadPoolTaskSupport
import scala.math.{exp, pow}

// Train a supervised Bayesian detector
object TrainBayesSuper {

  private val winSize = CVTools.makeSize(64, 128)
  private val winStride = CVTools.makeSize(0, 0)
  private val negWinStride = CVTools.makeSize(8, 8)
  private val blockSize = CVTools.makeSize(16, 16)
  private val blockStride = CVTools.makeSize(16, 16)
  private val cellSize = CVTools.makeSize(8, 8)


  // helper function to build an individual distribution for a set of images
  def trainLabel (images: Array[File], label: PASCALObjectLabel) (numBins: Int, numParts: Int)
      : (DirichletHashMapDist[DiscreteHOGCell], Int) = {

    var counter = 1
    val dist = new DirichletHashMapDist[DiscreteHOGCell](pow(numParts, numBins).toLong)

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling " + label + " training image " + counter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // normalized images are padded by 16 pixels on each side
      val cropBox = BoundingBox(Point(16, 16), 64, 128)
      val cropped = CVTools.cropImage(img, cropBox)

      // compute HOG descriptors
      val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)
      val discreteHOGs = descs.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts))

      dist.synchronized {
        counter += 1
        dist.addWordsMultiple(discreteHOGs, 100)
      }

    }

    // one window per image
    (dist, counter)

  }


  // helper function to mine an individual distribution from a set of negative images
  def trainNegative (images: Array[File]) (numBins: Int, numParts: Int)
      : (DirichletHashMapDist[DiscreteHOGCell], Int) = {

    var counter = 1
    var winCounter = 0
    val dist = new DirichletHashMapDist[DiscreteHOGCell](pow(numParts, numBins).toLong)

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("mining negative training image " + counter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)
      val discreteHOGs = descs.flatten.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts))

      dist.synchronized {
        counter += 1
        winCounter += descs.length
        dist.addWords(discreteHOGs)
      }

    }

    (dist, winCounter)

  }


  /* Train a detector. Arguments:
   * 0: -norm = train on normalized dataset or -full = train on original full images
   * 1: INRIA dataset location
   * 2: number of bins to use in HOG descriptor
   * 3: number of partitions to use for each gradient bin. this is used to take
   *    the space of HOG cell descriptors from a "continuous" to a discrete space
   * 4: file path for saving the learned detector
   *
   *    Note the total number of discetized HOG descriptors will be
   *    [num partitions]^[num gradient bins]
   */
  def main (args: Array[String]) : Unit = {

    CVTools.loadLibrary

    val inriaHome = args(1)
    val numBins = args(2).toInt
    val numParts = args(3).toInt
    val detectorFile = new File(args(4))

    val detectorOut = new ObjectOutputStream(new FileOutputStream(detectorFile))

    args(0) match {
      case "-norm" => {

        val posFolder = new File(inriaHome, "train_64x128_H96/pos")
        val negFolder = new File(inriaHome, "train_64x128_H96/neg")
        val posImages = posFolder.listFiles
        val negImages = negFolder.listFiles
        val prior = new HashMapDist[PASCALObjectLabel]

        val (posDist, numPos) = trainLabel(posImages, PASPerson)(numBins, numParts)
        posDist.display
        prior.addWordMultiple(PASPerson, numPos)

        val (negDist, numNeg) = trainNegative(negImages)(numBins, numParts)
        negDist.display
        prior.addWordMultiple(Negative, numNeg)

        prior.display

        val detector = new BayesHOGDetector(List(PASPerson, Negative), List(posDist, negDist), prior)
        detectorOut.writeObject(detector)

      }

      case "-full" => {

        val annotations = PASCALAnnotation.parseAnnotationList(inriaHome, "Train/annotations.lst")

        println("not yet implemented, but parse successful?")

      }

      case other =>
        throw new IllegalArgumentException("Unrecognized parameter: " + other)
    }

  }

}