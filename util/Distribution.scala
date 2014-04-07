package imgdetect.util

// trait for a generic probabilitity distribution
trait Distribution[T] extends Serializable {

  def conjugateProb (words: Array[T]) : Double
  def display : Unit
  def logConjugateProb (words: Array[T]) : Double
  def logProb (word: T) : Double
  def prob (word: T) : Double

}