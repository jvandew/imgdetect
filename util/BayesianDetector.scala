package imgdetect.util

trait BayesianDetector[T] extends Detector[T] {

  def detectLabelLogProp (words: Array[T], label: PASCALObjectLabel) : Double
  def detectLabelProp (words: Array[T], label: PASCALObjectLabel) : Double

  def detectLogProps (words: Array[T]) : List[(PASCALObjectLabel, Double)]
  def detectProps (words: Array[T]) : List[(PASCALObjectLabel, Double)]

  def likelihood (words: Array[T], label: PASCALObjectLabel) : Double
  def logLikelihood (words: Array[T], label: PASCALObjectLabel) : Double

  def logPrior (label: PASCALObjectLabel) : Double
  def prior (label: PASCALObjectLabel) : Double

}