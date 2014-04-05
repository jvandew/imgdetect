package imgdetect.util

import scala.collection.mutable.HashMap
import scala.math.log

/* A discrete probability distribution backed by a HashMap, this is most efficient for
 * representation of sparse distributions. To represent a dense distribution try using
 * an ArrayDist instead.
 */
class HashMapDist[T] extends DiscreteDistribution[T] {

  private val dist = new HashMap[T, Int]
  private var totalCount = 0L

  def addWord (word: T) : Unit = {

    dist.get(word) match {
      case None => dist(word) = 1
      case Some(num) => dist(word) += 1
    }

    totalCount += 1
  }

  def addWords (words: Array[T]) : Unit = {

    words.foreach { word =>
      dist.get(word) match {
        case None => dist(word) = 1
        case Some(num) => dist(word) += 1
      }
    }

    totalCount += words.length
  }

  // print out a visual representation of this distribution
  def display : Unit = {

    val keyDist = dist.toArray
    val sorted = keyDist.sortWith((kfreq1, kfreq2) => kfreq1._2 > kfreq2._2)
    sorted.foreach(kfreq => println(kfreq._1 + " : " + kfreq._2))

  }

  // TODO(jacob) become wise enough to know why this toDouble is required and not detected implicitly
  def logProb (word: T) : Double = log(dist.getOrElse(word, 0).toDouble) - log(totalCount)

  def numContained (word: T) : Int = dist.getOrElse(word, 0)

  def prob (word: T) : Double = dist.getOrElse(word, 0).toDouble / totalCount

  def totalSize: Long = totalCount

  def totalUnique: Int = dist.size

}