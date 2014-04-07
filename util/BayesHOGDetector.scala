package imgdetect.util

import scala.collection.immutable.HashMap
import scala.math.{exp, log}

// A BayesianDetector composed of a set of learned probability distributions
// on HOG features. Constructor takes a list of distributions and their labels.
class BayesHOGDetector (dists: List[(PASCALObjectLabel, DiscreteDistribution[DiscreteHOGCell])],
                           prior: DiscreteDistribution[PASCALObjectLabel])
  extends BayesianDetector[DiscreteHOGCell] {

  private val distMap = new HashMap[PASCALObjectLabel, DiscreteDistribution[DiscreteHOGCell]] ++ dists

  def this (labels: List[PASCALObjectLabel],
            dists: List[DiscreteDistribution[DiscreteHOGCell]],
            prior: DiscreteDistribution[PASCALObjectLabel]) = {

    this(labels.zip(dists), prior)
    require(labels.length == dists.length, "Number of labels and distributions must match")

  }


  // Perform detection on the given set of HOG descriptors. Returns a list of labels
  // and the probability of that label given hogCells
  def detect (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val propMap = distMap.map(ld => (ld._1, detectLabelProp(hogCells, ld._1)))
    val margLikelihood = propMap.values.reduce(_ + _)
    val probMap = propMap.map(lp => (lp._1, lp._2 / margLikelihood))

    probMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // Perform detection on the given set of HOG descriptors. Returns a list of labels
  // and the log probability of that label given hogCells
  def detectLog (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val logPropMap = distMap.map(ld => (ld._1, detectLabelLogProp(hogCells, ld._1)))
    val logMargLikelihood = log(logPropMap.values.reduce(exp(_) + exp(_)))
    val logProbMap = logPropMap.map(lp => (lp._1, lp._2 - logMargLikelihood))

    logProbMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // returns thelog  proportionality of the given HOG cells for the given label
  def detectLabelLogProp (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    logLikelihood(hogCells, label) + prior.logProb(label)


  // returns the proportionality of the given HOG cells for the given label
  def detectLabelProp (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    likelihood(hogCells, label) * prior.prob(label)


  // compute the likelihood of the given HOG cells under the given label
  def likelihood (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    distMap(label).conjugateProb(hogCells)

  // compute the log likelihood of the given HOG cell under the given label
  def logLikelihood (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    distMap(label).logConjugateProb(hogCells)


  def logPrior (label: PASCALObjectLabel) : Double = prior.logProb(label)


  def prior (label: PASCALObjectLabel) : Double = prior.prob(label)

}