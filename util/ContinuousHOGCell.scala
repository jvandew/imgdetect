package imgdetect.util

/* This class serves as a wrapper for an array of HOG cell gradients.
 * Gradients are stored as Floats to avoid inflating the memory footprint prior
 * to computation. Precision-wise this makes no difference. */
case class ContinuousHOGCell (val gradients: Array[Double]) extends VectorType {

  def this (grads: Array[Float]) = this(grads.map(_.toDouble))

  def toVector : Array[Double] = gradients

}
