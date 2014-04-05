package imgdetect.util

/* A helper class to represent a discrete version of a HOG descriptor cell.
 * numBins: the number of gradient bins for this HOG cell
 * numPartitions: the number of bins in each gradient bin. The total number
 * of possible unique DiscreteHOGCells will be [num partitions]^[num gradient bins]
 */
class DiscreteHOGCell (val numBins: Int, val numPartitions: Int, val gradients: Array[Int]) {

  if (gradients.length != numBins || !gradients.forall(_ < numPartitions)) {
    throw new IllegalArgumentException("Gradient array does not match given parameters")
  }

  def this (numBins: Int, numPartitions: Int) = this(numBins, numPartitions, new Array[Int](numBins))


  override def equals (other: Any) = other match {

    case hogObj: DiscreteHOGCell => numBins == hogObj.numBins &&
                                    numPartitions == hogObj.numPartitions &&
                                    gradients.sameElements(hogObj.gradients)
    case _ => false

  }


  // piggyback off String hashing
  override def hashCode : Int = {

    var hashStr = numBins.toString + numPartitions
    gradients.foreach(hashStr += _)

    hashStr.hashCode

  }


  def setGradients (grads: Array[Int]) : Unit = {

    if (grads.length != numBins || !grads.forall(_ < numPartitions)) {
      throw new IllegalArgumentException("Gradient array does not match given parameters")
    }
    else {
      grads.copyToArray(gradients)
    }

  }


  override def toString : String =
    "DiscreteHOGCell: " + numBins + " bins, " + numPartitions + " bin partitions, gradients = " + gradients

}