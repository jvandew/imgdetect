package imgdetect.util

import scala.math.{exp, log}

trait BayesianDetector[T] extends Detector[T] {

  // Perform detection on the given set of words. Returns a list of labels
  // and the probability of that label given words
  def detect (words: Array[T]) : List[(PASCALObjectLabel, Double)] = {

    val props = detectProps(words)
    val margLikelihood = props.foldLeft(0.0)((agg, lp) => agg + lp._2)
    props.map(lp => (lp._1, lp._2 / margLikelihood))

  }

  // Perform detection on the given set of words. Returns a list of labels
  // and the log probability of that label given words
  def detectLog (words: Array[T]) : List[(PASCALObjectLabel, Double)] = {

    val logProps = detectLogProps(words)
    val logMargLikelihood = log(logProps.foldLeft(0.0)((agg, lp) => agg + exp(lp._2)))
    logProps.map(lp => (lp._1, lp._2 - logMargLikelihood))

  }

  def detectLabelLogProp (words: Array[T], label: PASCALObjectLabel) : Double
  def detectLabelProp (words: Array[T], label: PASCALObjectLabel) : Double

  def detectLogProps (words: Array[T]) : List[(PASCALObjectLabel, Double)]
  def detectProps (words: Array[T]) : List[(PASCALObjectLabel, Double)]

  def likelihood (words: Array[T], label: PASCALObjectLabel) : Double
  def logLikelihood (words: Array[T], label: PASCALObjectLabel) : Double

  def logPrior (label: PASCALObjectLabel) : Double
  def prior (label: PASCALObjectLabel) : Double

}