package imgdetect.train

import imgdetect.cvtools.CVTools
import imgdetect.util.{BayesContHOGDetector, BayesDiscHOGDetector, BoundingBox,
                       DependentBayesHOGDetector, DirichletHashMapDist, DiscreteHOGCell,
                       HashMapDist, MultivarNormalDist, Negative, PASCALAnnotation,
                       PASCALObjectLabel, PASPerson, Point, Utils}
import java.io.{File, FileOutputStream, ObjectOutputStream}
import org.opencv.core.{Mat, Size}
import scala.collection.immutable.{HashMap, Map}
import scala.math.{exp, pow}

// Train a supervised Bayesian detector
object TrainBayesSuper {

  private val winSize = CVTools.makeSize(64, 128)
  private val winStride = CVTools.makeSize(0, 0)
  private val negWinStride = CVTools.makeSize(8, 8)
  private val blockSize = CVTools.makeSize(16, 16)
  private val blockStride = CVTools.makeSize(16, 16)
  private val cellSize = CVTools.makeSize(8, 8)


  // helper function to build a set of dependent distributions for a set of images
  def trainDepLabel (images: Array[File], label: PASCALObjectLabel) (numBins: Int, numParts: Int)
      : (Map[DiscreteHOGCell, DirichletHashMapDist[DiscreteHOGCell]],
         DirichletHashMapDist[DiscreteHOGCell],
         Int) = {

    def genDefaultDist = new DirichletHashMapDist[DiscreteHOGCell](pow(numParts, numBins).toLong)

    var counter = 1
    var deps = new HashMap[DiscreteHOGCell, DirichletHashMapDist[DiscreteHOGCell]].withDefaultValue(genDefaultDist)
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

      for (i <- 0 until discreteHOGs.length) {

        val depDist = deps.getOrElse(discreteHOGs(i), genDefaultDist)

        for (j <- 0 until discreteHOGs.length) {

          if (i != j) {
            // shouldn't be an expensive synchronization - probability of another
            // thread using this specific dist simultaneously is low
            depDist.synchronized {
              depDist.addWord(discreteHOGs(j))
            }
          }

        }

        this.synchronized {
          deps = deps.updated(discreteHOGs(i), depDist)
        }

      }

      // synchronized internally
      dist.addWords(discreteHOGs)

      this.synchronized {
        counter += 1
      }

    }

    // one window per image
    (deps, dist, counter)

  }


  // helper function to mine a set of dependent distributions from a set of negative images
  def trainDepNegative (images: Array[File]) (numBins: Int, numParts: Int)
      : (Map[DiscreteHOGCell, DirichletHashMapDist[DiscreteHOGCell]],
         DirichletHashMapDist[DiscreteHOGCell],
         Int) = {

    def genDefaultDist = new DirichletHashMapDist[DiscreteHOGCell](pow(numParts, numBins).toLong)

    var counter = 1
    var winCounter = 0
    var deps = new HashMap[DiscreteHOGCell, DirichletHashMapDist[DiscreteHOGCell]].withDefaultValue(genDefaultDist)
    val dist = new DirichletHashMapDist[DiscreteHOGCell](pow(numParts, numBins).toLong)

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("mining negative training image " + counter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)
      val discreteHOGs = descs.map(_.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts)))

      // iterate over each window
      for (w <- 0 until discreteHOGs.length) {
        for (i <- 0 until discreteHOGs(w).length) {

          val depDist = deps.getOrElse(discreteHOGs(w)(i), genDefaultDist)

          for (j <- 0 until discreteHOGs(w).length) {

            if (i != j) {
              // shouldn't be an expensive synchronization - probability of another
              // thread using this specific dist simultaneously is low
              depDist.synchronized {
                depDist.addWord(discreteHOGs(w)(j))
              }
            }

          }

          this.synchronized {
            deps = deps.updated(discreteHOGs(w)(i), depDist)
          }

        }
      }

      // synchronized internally
      dist.addWords(discreteHOGs.flatten)

      this.synchronized {
        counter += 1
        winCounter += descs.length
      }

    }

    (deps, dist, winCounter)

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
   * 1: -dep = train a dependent detector or -ind = train a detector with (invalid)
   *    assumptions about the independence of HOG cells in a detection window
   * 2: INRIA dataset location
   * 3: number of bins to use in HOG descriptor
   * 4: number of partitions to use for each gradient bin. this is used to take
   *    the space of HOG cell descriptors from a "continuous" to a discrete space
   * 5: file path for saving the learned detector
   *
   *    Note the total number of discetized HOG descriptors will be
   *    [num partitions]^[num gradient bins]
   */
  def main (args: Array[String]) : Unit = {

    CVTools.loadLibrary

    val inriaHome = args(2)
    val numBins = args(3).toInt
    val numParts = args(4).toInt
    val detectorFile = new File(args(5))

    val detectorOut = new ObjectOutputStream(new FileOutputStream(detectorFile))

    args(0) match {
      case "-norm" => {

        val posFolder = new File(inriaHome, "train_64x128_H96/pos")
        val negFolder = new File(inriaHome, "train_64x128_H96/neg")
        val posImages = posFolder.listFiles
        val negImages = negFolder.listFiles
        val prior = new HashMapDist[PASCALObjectLabel]

        val detector = args(1) match {
          case "-ind" => {

            val (posDist, numPos) = trainLabel(posImages, PASPerson)(numBins, numParts)
            posDist.display
            prior.addWordMultiple(PASPerson, numPos)

            val (negDist, numNeg) = trainNegative(negImages)(numBins, numParts)
            negDist.display
            prior.addWordMultiple(Negative, numNeg)

            // TODO(jacob) this really shouldn't be necessary
            posDist.scale(numNeg.toDouble / numPos)
            prior.display

            new BayesDiscHOGDetector(List(PASPerson, Negative), List(posDist, negDist), prior)
          }

          // note: pointless experiment
          case "-dep" => {

            val (posDeps, posDist, numPos) = trainDepLabel(posImages, PASPerson)(numBins, numParts)
            prior.addWordMultiple(PASPerson, numPos)

            val (negDeps, negDist, numNeg) = trainDepNegative(negImages)(numBins, numParts)
            prior.addWordMultiple(Negative, numNeg)

            prior.display

            new DependentBayesHOGDetector(List(PASPerson, Negative),
                                          List(posDeps, negDeps),
                                          List(posDist, negDist),
                                          prior)
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