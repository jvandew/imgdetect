package imgdetect.train

import imgdetect.cvtools.CVTools
import imgdetect.detector.{BayesContHOGDetector, BayesContLocationHOGDetector,
                           BayesDiscHOGDetector}
import imgdetect.prob.{DirichletHashMapDist, MultivarNormalDist}
import imgdetect.util.{BoundingBox, DiscreteHOGCell, Negative, PASCALAnnotation,
                       PASCALObjectLabel, PASPerson, Point, Utils}
import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import org.opencv.core.{Mat, Size}
import scala.collection.immutable.{HashMap, Map}
import scala.math.{exp, pow}
import scala.util.Random

// Train a supervised Bayesian detector
object TrainBayesSuper {

  private val winSize = CVTools.makeSize(64, 128)
  private val winStride = CVTools.makeSize(0, 0)
  private val negWinStride = CVTools.makeSize(8, 8)
  private val blockSize = CVTools.makeSize(16, 16)
  private val blockStride = CVTools.makeSize(16, 16)
  private val cellSize = CVTools.makeSize(8, 8)


  // helper function to build an individual distribution for a set of images
  def trainContLocLabel (images: Array[File], label: PASCALObjectLabel)
                        (numBins: Int)
                        (hogFiles: Array[File])
      : (Array[MultivarNormalDist], Int) = {

    var counter = 0
    val outs = hogFiles.map(f => new ObjectOutputStream(new FileOutputStream(f)))

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling " + label + " training image " + counter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // normalized images are padded by 16 pixels on each side
      val cropBox = BoundingBox(Point(16, 16), 64, 128)
      val cropped = CVTools.cropImage(img, cropBox)

      // compute HOG descriptors
      val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)
      val hogs = descs.flatten

      // write hog cells to their appropriate location files
      for (i <- 0 until hogs.length) {
        outs(i).synchronized {
          outs(i).writeObject(hogs(i))
          outs(i).reset
        }
      }

      this.synchronized {
        counter += 1
      }

    }

    outs.foreach(_.close)
    val ins = hogFiles.map(f => new ObjectInputStream(new FileInputStream(f)))

    var distCount = 0
    val dists = ins.map { in =>

      distCount += 1
      println("computing distribution " + distCount + " of " + ins.length)

      val hogs = Array.tabulate(counter)(_ => in.readObject.asInstanceOf[Array[Float]])
      MultivarNormalDist(hogs)

    }

    ins.foreach(_.close)

    // one window per image
    (dists, counter)

  }


  // helper function to mine an individual distribution from a set of negative images
  def trainContLocNegative (images: Array[File]) (numBins: Int) (hogFiles: Array[File])
      : (Array[MultivarNormalDist], Int) = {

    var counter = 1
    var winCounter = 0
    val rand = new Random
    val outs = hogFiles.map(f => new ObjectOutputStream(new FileOutputStream(f)))

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("mining negative training image " + counter + " of " + images.length + ":\n\t" + path )

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)
      val winHOGs = descs.map(_.flatten)

      // write hog cells to their appropriate location files
      winHOGs.foreach { hogs =>
        for (i <- 0 until hogs.length) {
          outs(i).synchronized {
            outs(i).writeObject(hogs(i))
            outs(i).reset
          }
        }
      }

      this.synchronized {
        counter += 1
        winCounter += descs.length
      }

    }

    outs.foreach(_.close)
    val ins = hogFiles.map(f => new ObjectInputStream(new FileInputStream(f)))

    var distCount = 0
    val dists = ins.map { in =>

      distCount += 1
      println("computing distribution " + distCount + " of " + ins.length)

      val hogs = Array.tabulate(winCounter)(_ => in.readObject.asInstanceOf[Array[Float]])
      MultivarNormalDist(hogs)

    }

    ins.foreach(_.close)

    (dists, winCounter)

  }


  // helper function to build an individual distribution for a set of images
  def trainContLabel (images: Array[File], label: PASCALObjectLabel) (numBins: Int)
      : (MultivarNormalDist, Int) = {

    var counter = 0

    // positive images can fit in-memory
    val hogs = images.par.map { imgFile =>

      val path = imgFile.getPath

      println("handling " + label + " training image " + counter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // normalized images are padded by 16 pixels on each side
      val cropBox = BoundingBox(Point(16, 16), 64, 128)
      val cropped = CVTools.cropImage(img, cropBox)

      // compute HOG descriptors
      val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

      this.synchronized {
        counter += 1
      }

      descs.flatten.map(_.map(_.toDouble))

    }

    // one window per image
    (MultivarNormalDist(hogs.toArray.flatten), counter)

  }


  // helper function to mine an individual distribution from a set of negative images
  def trainContNegative (images: Array[File]) (numBins: Int) (propData: Double)
      : (MultivarNormalDist, Int) = {

    val rand = new Random
    var counter = 1
    var winCounter = 0

    val hogs = images.par.map { imgFile =>

      val path = imgFile.getPath

      println("mining negative training image " + counter + " of " + images.length + ":\n\t" + path )

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)

      this.synchronized {
        counter += 1
        winCounter += descs.length
      }

      // holy functions Batman
      // flatten to an array of HOGs, take our subset, then convert HOGs to Doubles
      descs.flatten.flatten.filter(_ => rand.nextDouble < propData).map(_.map(_.toDouble))

    }

    val flatHOGs = hogs.toArray.flatten
    println("Building a distribution on " + flatHOGs.length + " HOG cells")

    (MultivarNormalDist(flatHOGs), winCounter)

  }


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
      dist.addWords(discreteHOGs)

      this.synchronized {
        counter += 1
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
      dist.addWords(discreteHOGs)

      this.synchronized {
        counter += 1
        winCounter += descs.length
      }

    }

    (dist, winCounter)

  }


  /* Train a detector. Arguments:
   * 0: -norm = train on normalized dataset or -full = train on original full images
   * 1: -disc = train a discrete detector
   *    -cont = train a continuous detector
   *    -contLoc = train a location-aware continuous detector
   * 2: INRIA dataset location
   * 3: file path for saving the learned detector
   * 4: number of bins to use in HOG descriptor
   * 5: -disc case: number of partitions to use for each gradient bin. this is used
   *    to take the space of HOG cell descriptors from a "continuous" to a discrete
   *    space
   *    -cont case: the proportion of negative training data to use. as the dataset
   *    can be too large to fit in memory (necessary for computing covariances) this
   *    provides a method for making training possible
   *    -contLoc case: no parameter required
   *
   *    Note the total number of discetized HOG descriptors will be
   *    [num partitions]^[num gradient bins]
   */
  def main (args: Array[String]) : Unit = {

    CVTools.loadLibrary

    val inriaHome = args(2)
    val numBins = args(4).toInt

    val detectorOut = new ObjectOutputStream(new FileOutputStream(args(3)))

    args(0) match {
      case "-norm" => {

        val posFolder = new File(inriaHome, "train_64x128_H96/pos")
        val negFolder = new File(inriaHome, "train_64x128_H96/neg")
        val posImages = posFolder.listFiles
        val negImages = negFolder.listFiles
        val prior = new DirichletHashMapDist[PASCALObjectLabel](2)

        val detector = args(1) match {
          case "-cont" => {

            val propData = args(5).toDouble

            val (posDist, numPos) = trainContLabel(posImages, PASPerson)(numBins)
            prior.addWordMultiple(PASPerson, numPos)

            val (negDist, numNeg) = trainContNegative(negImages)(numBins)(propData)
            prior.addWordMultiple(Negative, numNeg)

            prior.display

            new BayesContHOGDetector(List(PASPerson, Negative), List(posDist, negDist), prior)
          }

          case "-contLoc" => {

            val vectorLength = ((winSize.width / cellSize.width) * (winSize.height / cellSize.height)).toInt

            val hogDir = File.createTempFile("hogs", "dir", new File("."))
            hogDir.delete
            hogDir.mkdir
            hogDir.deleteOnExit
            val hogFiles = Array.tabulate(vectorLength)(i => File.createTempFile("hog_" + i, "loc", hogDir))

            val (posDists, numPos) = trainContLocLabel(posImages, PASPerson)(numBins)(hogFiles)
            prior.addWordMultiple(PASPerson, numPos)

            val (negDists, numNeg) = trainContLocNegative(negImages)(numBins)(hogFiles)
            prior.addWordMultiple(Negative, numNeg)

            prior.display

            new BayesContLocationHOGDetector(List(PASPerson, Negative),
                                             List(posDists, negDists),
                                             prior,
                                             vectorLength)
          }

          case "-disc" => {

            val numParts = args(5).toInt

            val (posDist, numPos) = trainLabel(posImages, PASPerson)(numBins, numParts)
            posDist.display
            prior.addWordMultiple(PASPerson, numPos)

            val (negDist, numNeg) = trainNegative(negImages)(numBins, numParts)
            negDist.display
            prior.addWordMultiple(Negative, numNeg)

            // TODO(jacob) this really shouldn't be necessary, should it?
            posDist.scale(numNeg.toDouble / numPos)
            prior.display

            new BayesDiscHOGDetector(List(PASPerson, Negative), List(posDist, negDist), prior)
          }
        }

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