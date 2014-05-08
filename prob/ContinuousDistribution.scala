package imgdetect.prob

// trait for a continuous probability distribution
trait ContinuousDistribution[T] extends Distribution[T] {

  // get the expected value, or mean, of the distribution
  def mean () : T

}
