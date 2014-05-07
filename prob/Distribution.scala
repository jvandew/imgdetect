package imgdetect.prob

// trait for a generic probabilitity distribution
trait Distribution[T] extends Serializable {

  def display () : Unit
  def logProb (word: T) : Double
  def prob (word: T) : Double

  /* Assumes independence of words.
   * Subclasses should override these methods if this is not the case.
   */
  def conjugateProb (words: Array[T]) : Double = words.map(prob(_)).fold(1.0)(_ * _)
  def logConjugateProb (words: Array[T]) : Double = words.map(logProb(_)).fold(0.0)(_ + _)

}