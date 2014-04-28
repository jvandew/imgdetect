package imgdetect.util

/* This class serves as a wrapper for an array of HOG cell gradients.
 * Gradients are stored as Floats to avoid inflating the memory footprint prior
 * to computation. Precision-wise this makes no difference. */
case class ContinuousHOGCell (val gradients: Array[Float]) extends VectorType {

  def this (grads: Array[Double]) = this(grads.map(_.toFloat))

  def toVector : Array[Double] = gradients.map(_.toDouble)

}
