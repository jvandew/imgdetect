package imgdetect.util

import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.stat.correlation.Covariance
import scala.math.log

/* Companion object to make our lives simpler */
object MultivarNormalDist {

  def apply (data: Array[Array[Double]]) : MultivarNormalDist = {

    // this is an inefficient operation cache-wise
    val sums = data.reduce((v1, v2) => Array.tabulate(v1.length)(i => v1(i) + v2(i)))
    val means = sums.map(_ / data.length)
    val covarMatrix = new Covariance(data).getCovarianceMatrix.getData

    new MultivarNormalDist(means, covarMatrix)

  }

  def apply (data: Array[Array[Float]]) : MultivarNormalDist =
    apply(data.map(_.map(_.toDouble)))

}

/* A wrapper class around the Apache Commons MultivariateNormalDistribution. */
class MultivarNormalDist (means: Array[Double], covarMatrix: Array[Array[Double]])
  extends ContinuousDistribution[Array[Double]] {

  /* This value is lazy by necessity, as MultivariateNormalDistribution is not Serializable
   * for some ungodly reason. This also means that MultivarNormalDists MUST NOT be used
   * to compute probabilities prior to serialization (display should be ok).
   */
  private lazy val dist = new MultivariateNormalDistribution(means, covarMatrix)

  // print out a visual representation of this distribution
  def display () : Unit = {
    println("Mean: " + means.mkString("(", ", ", ")"))
    println("Covariance Matrix:\n" + covarMatrix.mkString("\t|", "\t", "|").mkString("\n"))
  }

  def logProb (vector: Array[Double]) : Double = log(dist.density(vector))

  def prob (vector: Array[Double]) : Double = dist.density(vector)

}