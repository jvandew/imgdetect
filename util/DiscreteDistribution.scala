package imgdetect.util

// abstract class for a discrete probability distribution
abstract class DiscreteDistribution[T] extends Distribution[T] {

  def addWord (word: T) : Unit
  def addWordMultiple (word: T, num: Int) : Unit
  def addWords (words: Array[T]) : Unit
  def addWordsMultiple (words: Array[T], num: Int) : Unit
  def numContained (word: T) : Int
  def totalSize: Long
  def totalUnique: Long

  def conjugateProb (words: Array[T]) : Double = words.map(prob(_)).reduce(_ * _)
  def logConjugateProb (words: Array[T]) : Double = words.map(logProb(_)).reduce(_ + _)

}