package imgdetect.util

import scala.collection.immutable.{HashMap, Map}
import scala.math.{exp, log}

// A BayesianDetector composed of a set of learned dependent probability distributions
// on HOG features. Constructor takes a list of labels and a hash map for each
// mapping a HOG cell to a distribution of HOG cells conditional on this cell.
class DependentBayesHOGDetector (deps: List[(PASCALObjectLabel, Map[DiscreteHOGCell, DiscreteDistribution[DiscreteHOGCell]])],
                        dists: List[(PASCALObjectLabel, DiscreteDistribution[DiscreteHOGCell])],
                        prior: DiscreteDistribution[PASCALObjectLabel])
  extends BayesianDetector[DiscreteHOGCell] {

  private val depMap = new HashMap[PASCALObjectLabel, Map[DiscreteHOGCell, DiscreteDistribution[DiscreteHOGCell]]] ++ deps
  private val distMap = new HashMap[PASCALObjectLabel, DiscreteDistribution[DiscreteHOGCell]] ++ dists

  def this (labels: List[PASCALObjectLabel],
            deps: List[Map[DiscreteHOGCell, DiscreteDistribution[DiscreteHOGCell]]],
            dists: List[DiscreteDistribution[DiscreteHOGCell]],
            prior: DiscreteDistribution[PASCALObjectLabel]) = {

    this(labels.zip(deps), labels.zip(dists), prior)
    require(labels.length == dists.length && labels.length == deps.length,
            "Number of labels and distributions must match")

  }


  // Perform detection on the given set of HOG descriptors. Returns a list of labels
  // and the probability of that label given hogCells
  def detect (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val props = detectProps(hogCells)
    val margLikelihood = props.foldLeft(0.0)((agg, lp) => agg + lp._2)
    props.map(lp => (lp._1, lp._2 / margLikelihood))

  }


  // Perform detection on the given set of HOG descriptors. Returns a list of labels
  // and the log probability of that label given hogCells
  def detectLog (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val logProps = detectLogProps(hogCells)
    val logMargLikelihood = log(logProps.foldLeft(0.0)((agg, lp) => agg + exp(lp._2)))
    logProps.map(lp => (lp._1, lp._2 - logMargLikelihood))

  }


  // Compute the log proportionality of each label for the given set of HOG
  // descriptors. Returns a list of labels and the log proportionality of each.
  def detectLogProps (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val logPropMap = distMap.map(ld => (ld._1, detectLabelLogProp(hogCells, ld._1)))
    logPropMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // Compute the proportionality of each label for the given set of HOG
  // descriptors. Returns a list of labels and the proportionality of each.
  def detectProps (hogCells: Array[DiscreteHOGCell]) : List[(PASCALObjectLabel, Double)] = {

    val propMap = distMap.map(ld => (ld._1, detectLabelProp(hogCells, ld._1)))
    propMap.toArray.sortWith((lp1, lp2) => lp1._2 > lp2._2).toList

  }


  // returns the log proportionality of the given HOG cells for the given label
  def detectLabelLogProp (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    logLikelihood(hogCells, label) + prior.logProb(label)


  // returns the proportionality of the given HOG cells for the given label
  def detectLabelProp (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double =
    likelihood(hogCells, label) * prior.prob(label)


  // compute the likelihood of the given HOG cells under the given label
  def likelihood (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double = {

    var like = distMap(label).prob(hogCells(0))
    for (i <- 1 until hogCells.length) {
      like *= depMap(label)(hogCells(i-1)).prob(hogCells(i))
    }

    like

  }

  // compute the log likelihood of the given HOG cells under the given label
  def logLikelihood (hogCells: Array[DiscreteHOGCell], label: PASCALObjectLabel) : Double = {

    var like = distMap(label).logProb(hogCells(0))
    for (i <- 1 until hogCells.length) {
      like += depMap(label)(hogCells(i-1)).logProb(hogCells(i))
    }

    like

  }


  def logPrior (label: PASCALObjectLabel) : Double = prior.logProb(label)


  def prior (label: PASCALObjectLabel) : Double = prior.prob(label)

}