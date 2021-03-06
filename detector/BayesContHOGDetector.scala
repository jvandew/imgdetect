package imgdetect.detector

import imgdetect.prob.{ContinuousDistribution, DiscreteDistribution}
import imgdetect.util.PASCALObjectLabel
import scala.collection.immutable.HashMap
import scala.math.{exp, log}

// A BayesianDetector composed of a set of learned Gaussian distributions
// on HOG features. Constructor takes a list of Gaussians and their labels.
// This detector still takes a discrete prior distribution
class BayesContHOGDetector (dists: List[(PASCALObjectLabel, ContinuousDistribution[Array[Double]])],
                            prior: DiscreteDistribution[PASCALObjectLabel])
  extends BayesianDetector[Array[Double]] {

  val distMap = new HashMap[PASCALObjectLabel, ContinuousDistribution[Array[Double]]] ++ dists

  def this (labels: List[PASCALObjectLabel],
            dists: List[ContinuousDistribution[Array[Double]]],
            prior: DiscreteDistribution[PASCALObjectLabel]) = {

    this(labels.zip(dists), prior)
    require(labels.length == dists.length, "Number of labels and distributions must match")

  }


  // Compute the log proportionality of each label for the given set of HOG
  // descriptors. Returns a list of labels and the log proportionality of each.
  def detectLogProps (hogCells: Array[Array[Double]]) : List[(PASCALObjectLabel, Double)] = {

    val logPropMap = distMap.map(ld => (ld._1, detectLabelLogProp(hogCells, ld._1)))
    logPropMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // Compute the proportionality of each label for the given set of HOG
  // descriptors. Returns a list of labels and the proportionality of each.
  def detectProps (hogCells: Array[Array[Double]]) : List[(PASCALObjectLabel, Double)] = {

    val propMap = distMap.map(ld => (ld._1, detectLabelProp(hogCells, ld._1)))
    propMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // returns the log proportionality of the given HOG cells for the given label
  def detectLabelLogProp (hogCells: Array[Array[Double]], label: PASCALObjectLabel) : Double =
    logLikelihood(hogCells, label) + prior.logProb(label)


  // returns the proportionality of the given HOG cells for the given label
  def detectLabelProp (hogCells: Array[Array[Double]], label: PASCALObjectLabel) : Double =
    likelihood(hogCells, label) * prior.prob(label)


  // compute the likelihood of the given HOG cells under the given label
  def likelihood (hogCells: Array[Array[Double]], label: PASCALObjectLabel) : Double =
    distMap(label).conjugateProb(hogCells)

  // compute the log likelihood of the given HOG cell under the given label
  def logLikelihood (hogCells: Array[Array[Double]], label: PASCALObjectLabel) : Double =
    distMap(label).logConjugateProb(hogCells)


  def logPrior (label: PASCALObjectLabel) : Double = prior.logProb(label)


  def prior (label: PASCALObjectLabel) : Double = prior.prob(label)

}