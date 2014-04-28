package imgdetect.util

import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.stat.correlation.Covariance
import scala.math.log

/* A wrapper class around the Apache Commons MultivariateNormalDistribution.
 * This takes a function for implicit conversion from the word type to an array
 * representing a vector in multi-dimensional space.
 */
class MultivarNormalDist[T <: VectorType] (data: Array[T])
  extends ContinuousDistribution[T] {

  private val doubleData = data.map(_.toVector)
  private val means = doubleData.map(ds => ds.fold(0.0)(_ + _) / ds.length)
  private val covarMatrix = new Covariance(doubleData).getCovarianceMatrix.getData

  private val dist = new MultivariateNormalDistribution(means, covarMatrix)


  // print out a visual representation of this distribution
  def display () : Unit = {
    println("Mean: " + means.mkString("(", ", ", ")"))
    println("Covariance Matrix:\n" + covarMatrix.mkString("\t|", "\t", "|").mkString("\n"))
  }

  def logProb (word: T) : Double = log(dist.density(word.toVector))

  def prob (word: T) : Double = dist.density(word.toVector)

}