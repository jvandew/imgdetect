package imgdetect.prob

import scala.math.log

/* A discrete probability distribution backed by an array, this is most efficient for
 * representation of dense distributions. To represent a sparse distribution try using
 * a HashMapDist instead.
 */
class ArrayDist[T] (computeIndex: T => Int, invertIndex: Int => T, maxIndex: Int) extends DiscreteDistribution[T] {

  private val dist = new Array[Int](maxIndex)
  private var totalCount = 0L

  def addWord (word: T) : Unit = {
    dist(computeIndex(word)) += 1
    totalCount += 1
  }

  def addWordMultiple (word: T, num: Int) : Unit = {
    dist(computeIndex(word)) += num
    totalCount += num
  }

  def addWords (words: Array[T]) : Unit = {
    words.foreach(word => dist(computeIndex(word)) += 1)
    totalCount += words.length
  }

  def addWordsMultiple (words: Array[T], num: Int) : Unit = {
    words.foreach(word => dist(computeIndex(word)) += num)
    totalCount += words.length * num
  }

  // print out a visual representation of this distribution
  def display () : Unit = {

    val indexDist = dist.zipWithIndex
    val sorted = indexDist.sortWith((ifreq1, ifreq2) => ifreq1._2 < ifreq2._2)
    sorted.foreach(ifreq => println(invertIndex(ifreq._1) + " : " + ifreq._2))

  }

  def logProb (word: T) : Double = log(computeIndex(word)) - log(totalCount)

  def numContained (word: T) : Long = dist(computeIndex(word)).toLong

  def prob (word: T) : Double = dist(computeIndex(word)).toDouble / totalCount

  def totalSize: Long = totalCount

  def totalUnique: Long = dist.count(_ != 0)

}