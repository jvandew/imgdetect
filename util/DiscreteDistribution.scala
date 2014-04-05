package imgdetect.util

// abstract class for a discrete probability distribution
abstract class DiscreteDistribution[T] {

  def addWord (word: T) : Unit
  def logProb (word: T) : Double
  def numContained (word: T) : Int
  def prob (word: T) : Double
  def totalSize: Long

}