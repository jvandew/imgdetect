package imgdetect.util

import org.apache.commons.math3.special.Gamma
import scala.collection.mutable.HashMap
import scala.math.log

/* A Dirichlet distribution backed by a HashMap. */
class DirichletHashMapDist[T] (numCategories: Long) extends DiscreteDistribution[T] {

  private val dist = new HashMap[T, Long].withDefaultValue(1L)
  private var totalCount = numCategories

  def addWord (word: T) : Unit = {

    dist(word) += 1
    totalCount += 1
  }

  def addWordMultiple (word: T, num: Int) : Unit = {

    dist(word) += num
    totalCount += num
  }

  def addWords (words: Array[T]) : Unit = {

    // TODO(jacob) make entire class thread-safe
    words.foreach(w => dist.synchronized(dist(w) += 1))
    this.synchronized(totalCount += words.length)
  }

  def addWordsMultiple (words: Array[T], num: Int) : Unit = {

    words.foreach(dist(_) += num)
    totalCount += words.length * num
  }

  override def conjugateProb (words: Array[T]) = {

    val wordMap = new HashMap[T, Int].withDefaultValue(0)
    words.foreach(wordMap(_) += 1)

    val gammaProd = wordMap.foldLeft(1.0) { (agg, kv) =>
      agg * Gamma.gamma(dist(kv._1) + kv._2) / Gamma.gamma(dist(kv._1))
    }
    Gamma.gamma(totalCount) * gammaProd / Gamma.gamma(totalCount + words.length)
  }

  // print out a visual representation of this distribution
  def display () : Unit = {

    val keyDist = dist.toArray
    val sorted = keyDist.sortWith((kfreq1, kfreq2) => kfreq1._2 < kfreq2._2)
    sorted.foreach(kfreq => println(kfreq._1 + " : " + kfreq._2))

  }

  override def logConjugateProb (words: Array[T]) = {

    val wordMap = new HashMap[T, Int].withDefaultValue(0)
    words.foreach(wordMap(_) += 1)

    val gammaProd = wordMap.foldLeft(0.0) { (agg, kv) =>
      agg + Gamma.logGamma(dist(kv._1) + kv._2) - Gamma.logGamma(dist(kv._1))
    }

    Gamma.logGamma(totalCount) + gammaProd - Gamma.logGamma(totalCount + words.length)

  }

  def logProb (word: T) : Double =
    Gamma.logGamma(totalCount) + Gamma.logGamma(1 + dist(word)) - Gamma.logGamma(totalCount + 1)

  def numContained (word: T) : Long = dist(word)

  def prob (word: T) : Double =
    Gamma.gamma(totalCount) * Gamma.gamma(1 + dist(word)) / Gamma.gamma(totalCount + 1)

  // scale Dirichlet parameters by a constant factor
  def scale (factor: Double) : Unit = {

    dist.foreach(kv => dist(kv._1) = (kv._2 * factor).toLong)
    totalCount = dist.foldLeft(0L)((agg, kv) => agg + kv._2)

  }

  def totalSize: Long = totalCount

  def totalUnique: Long = numCategories

}