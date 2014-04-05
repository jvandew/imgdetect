package imgdetect.train

import imgdetect.util.{PASCALAnnotation, Utils}

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

    val inriaHome = args(1)
    val numBins = args(2).toInt
    val numParts = args(3).toInt

    args(0) match {
      case "-norm" => {

        ()

      }

      case "-full" => {

        val annotations = PASCALAnnotation.parseAnnotationList(inriaHome, annoFilename)

        println("parse successful?")

      }

      case other =>
        throw new IllegalArgumentException("Unrecognized parameter: " + other)
    }

  }

}