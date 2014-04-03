package imgdetect.train

import imgdetect.util.{PASCALAnnotation, Utils}

// Train a supervised Bayesian detector
object TrainBayesSuper {

  /* Train a detector. Arguments:
   * 0: INRIA dataset location
   * 1: annotation file list (relative to INRIA main folder)
   */
  def main (args: Array[String]) : Unit = {

    val inriaHome = args(0)
    val annoFilename = args(1)
    val annotations = PASCALAnnotation.parseAnnotationList(inriaHome, annoFilename)

    println("parse successful?")

  }

}