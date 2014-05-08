package imgdetect.test

import imgdetect.cvtools.CVTools
import imgdetect.detector.{BayesContHOGDetector, BayesContLocationHOGDetector,
                           BayesDiscHOGDetector, BayesianDetector}
import imgdetect.util.{BoundingBox, DiscreteHOGCell, Negative, PASCALObjectLabel, PASPerson, Point}
import java.io.{File, FileInputStream, ObjectInputStream}

// Test a supervised Bayesian detector
object TestBayesSuper {

  private val winSize = CVTools.makeSize(64, 128)
  private val winStride = CVTools.makeSize(0, 0)
  private val negWinStride = CVTools.makeSize(8, 8)
  private val blockSize = CVTools.makeSize(16, 16)
  private val blockStride = CVTools.makeSize(16, 16)
  private val cellSize = CVTools.makeSize(8, 8)


  // helper function to test positive examples; returns the number of true positives and false negatives
  def testContPositive (images: Array[File])
                       (label: PASCALObjectLabel)
                       (detector: BayesianDetector[Array[Double]])
                       (numBins: Int)
      : (Int, Int) = {

    var imgCounter = 1
    var tp = 0
    var fn = 0

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling positive test image " + imgCounter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // normalized test images are padded by 3 pixels on each side
      val cropBox = BoundingBox(Point(3, 3), 64, 128)
      val cropped = CVTools.cropImage(img, cropBox)

      // compute HOG descriptors
      val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

      val results = detector.detectLogProps(descs.flatten.map(_.map(_.toDouble)))
      results.foreach(lp => println("\t" + lp._1 + ": " + lp._2))

      this.synchronized {

        imgCounter += 1

        results match {
          case (`label`, _)::_ => tp += 1
          case (Negative, _)::_ => fn += 1
          case _ => throw new Exception("life fail")
        }

      }

    }

    println("\nOut of " + images.length + " images, " + tp + " true positives and " + fn + " false negatives")

    (tp, fn)

  }


  // Helper function to test negative examples; returns the number of true negatives and false positives.
  // If an image window scores higher than Negative on any label this is treated as a false positive.
  def testContNegative (images: Array[File])
                       (detector: BayesianDetector[Array[Double]])
                       (numBins: Int)
      : (Int, Int) = {

    var imgCounter = 1
    var tn = 0
    var fp = 0

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling negative test image " + imgCounter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)
      val imgHOGs = descs.map(_.flatten)

      var tnWins = 0
      var fpWins = 0

      imgHOGs.foreach { hogs =>
        detector.detectLogProps(hogs.map(_.map(_.toDouble))) match {
          case (Negative, _)::_ => tnWins += 1
          case _::_ => fpWins += 1
          case _ => throw new Exception("life fail")
        }
      }

      println("True Negatives: " + tnWins + "\nFalse Positives: " + fpWins)

      this.synchronized {
        imgCounter += 1
        tn += tnWins
        fp += fpWins
      }

    }

    println("\nOut of " + (tn + fp) + " windows, " + tn + " true negatives and " + fp + " false positives")

    (tn, fp)

  }


  // helper function to test positive examples; returns the number of true positives and false negatives
  def testDiscPositive (images: Array[File])
                       (label: PASCALObjectLabel)
                       (detector: BayesianDetector[DiscreteHOGCell])
                       (numBins: Int, numParts: Int)
      : (Int, Int) = {

    var imgCounter = 1
    var tp = 0
    var fn = 0

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling positive test image " + imgCounter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // normalized test images are padded by 3 pixels on each side
      val cropBox = BoundingBox(Point(3, 3), 64, 128)
      val cropped = CVTools.cropImage(img, cropBox)

      // compute HOG descriptors
      val descs = CVTools.computeHOGInFullImage(cropped)(winSize, winStride, blockSize, blockStride, cellSize, numBins)
      val discreteHOGs = descs.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts))

      val results = detector.detectLogProps(discreteHOGs)
      results.foreach(lp => println("\t" + lp._1 + ": " + lp._2))

      this.synchronized {

        imgCounter += 1

        results match {
          case (`label`, _)::_ => tp += 1
          case (Negative, _)::_ => fn += 1
          case _ => throw new Exception("life fail")
        }

      }

    }

    println("\nOut of " + images.length + " images, " + tp + " true positives and " + fn + " false negatives")

    (tp, fn)

  }


  // Helper function to test negative examples; returns the number of true negatives and false positives.
  // If an image window scores higher than Negative on any label this is treated as a false positive.
  def testDiscNegative (images: Array[File])
                       (detector: BayesianDetector[DiscreteHOGCell])
                       (numBins: Int, numParts: Int)
      : (Int, Int) = {

    var imgCounter = 1
    var tn = 0
    var fp = 0

    images.par.foreach { imgFile =>

      val path = imgFile.getPath

      println("handling negative test image " + imgCounter + " of " + images.length + ":\n\t" + path)

      val img = CVTools.imreadGreyscale(path)

      // compute HOG descriptors
      val descs = CVTools.computeHOGWindows(img)(winSize, negWinStride, blockSize, blockStride, cellSize, numBins)
      val discreteHOGs = descs.map(_.flatten.map(DiscreteHOGCell.discretizeHOGCell(_, numParts)))

      var tnWins = 0
      var fpWins = 0

      discreteHOGs.foreach { hogs =>
        detector.detectLogProps(hogs) match {
          case (Negative, _)::_ => tnWins += 1
          case _::_ => fpWins += 1
          case _ => throw new Exception("life fail")
        }
      }

      println("True Negatives: " + tnWins + "\nFalse Positives: " + fpWins)

      this.synchronized {
        imgCounter += 1
        tn += tnWins
        fp += fpWins
      }

    }

    println("\nOut of " + (tn + fp) + " windows, " + tn + " true negatives and " + fp + " false positives")

    (tn, fp)

  }


  /* Test a detector. Arguments:
   * 0: file path for saving the learned detector
   * 1: INRIA dataset location
   * 2: number of bins to use in HOG descriptor
   * 3: number of partitions to use for each gradient bin. this is used to take
   *    the space of HOG cell descriptors from a "continuous" to a discrete space
   *    (discrete detector only)
   */
  def main (args: Array[String]) : Unit = {

    CVTools.loadLibrary

    val inriaHome = args(1)
    val numBins = args(2).toInt

    val posFolder = new File(inriaHome, "test_64x128_H96/pos")
    val negFolder = new File(inriaHome, "test_64x128_H96/neg")
    val posImages = posFolder.listFiles
    val negImages = negFolder.listFiles

    val detectorIn = new ObjectInputStream(new FileInputStream(args(0)))

    detectorIn.readObject match {
      case contDet: BayesContHOGDetector => {

        println("Testing continuous Bayesian detector...\n")

        testContPositive(posImages)(PASPerson)(contDet)(numBins)
        testContNegative(negImages)(contDet)(numBins)
      }

      case contLocDet: BayesContLocationHOGDetector => {

        println("Testing continuous location-aware Bayesian detector...\n")

        testContPositive(posImages)(PASPerson)(contLocDet)(numBins)
        testContNegative(negImages)(contLocDet)(numBins)
      }

      case discDet: BayesDiscHOGDetector => {

        val numParts = args(3).toInt
        println("Testing discrete Bayesian detector...\n")

        testDiscPositive(posImages)(PASPerson)(discDet)(numBins, numParts)
        testDiscNegative(negImages)(discDet)(numBins, numParts)
      }

      case _ =>
        throw new IllegalArgumentException("A detector could not be parsed from " + args(0))

    }
  }

}