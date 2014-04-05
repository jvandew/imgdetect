package imgdetect.util

import scala.math.log

/* A discrete probability distribution backed by an array, this is most efficient for
 * representation of dense distributions. To represent a sparse distribution try using
 * a HashMapDist instead.
 */
class ArrayDist[T] (computeIndex: T => Int, maxIndex: Int) extends DiscreteDistribution[T] {

  private val dist = new Array[Int](maxIndex)
  private var totalCount = 0L

  def addWord (word: T) : Unit = {
    dist(computeIndex(word)) += 1
    totalCount += 1
  }

  def logProb (word: T) : Double = log(computeIndex(word)) - log(totalCount)

  def numContained (word: T) : Int = dist(computeIndex(word))

  def prob (word: T) : Double = dist(computeIndex(word)).toDouble / totalCount

  def totalSize: Long = totalCount

}