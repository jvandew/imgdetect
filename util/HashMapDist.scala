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

  // TODO(jacob) become wise enough to know why this toDouble is required and not detected implicitly
  def logProb (word: T) : Double = log(dist.getOrElse(word, 0).toDouble) - log(totalCount)

  def numContained (word: T) : Int = dist.getOrElse(word, 0)

  def prob (word: T) : Double = dist.getOrElse(word, 0).toDouble / totalCount

  def totalSize: Long = totalCount

}