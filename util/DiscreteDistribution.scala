package imgdetect.util

// abstract class for a discrete probability distribution
trait DiscreteDistribution[T] extends Distribution[T] {

  def addWord (word: T) : Unit
  def addWordMultiple (word: T, num: Int) : Unit
  def addWords (words: Array[T]) : Unit
  def addWordsMultiple (words: Array[T], num: Int) : Unit
  def numContained (word: T) : Long
  def totalSize: Long
  def totalUnique: Long

}