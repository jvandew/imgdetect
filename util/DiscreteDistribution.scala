package imgdetect.util

// abstract class for a discrete probability distribution
abstract class DiscreteDistribution[T] {

  def addWord (word: T) : Unit
  def addWords (words: Array[T]) : Unit
  def display : Unit
  def logProb (word: T) : Double
  def numContained (word: T) : Int
  def prob (word: T) : Double
  def totalSize: Long
  def totalUnique: Int

}