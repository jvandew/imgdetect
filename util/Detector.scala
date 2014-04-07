package imgdetect.util

trait Detector[T] extends Serializable {

  def detect (words: Array[T]) : List[(PASCALObjectLabel, Double)]
  def detectLog (hogCells: Array[T]) : List[(PASCALObjectLabel, Double)]

}