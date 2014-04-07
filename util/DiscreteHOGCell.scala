package imgdetect.util

/* Companion class for DiscreteHOGCell */
object DiscreteHOGCell {

  // convert a cell of continuous HOG gradients to a DiscreteHOGCell
  // TODO(jacob) I believe it's technically possible for floating point errors
  //     to cause (g * numParts).toInt == numParts, an error condition
  def discretizeHOGCell (grads: Array[Float], numParts: Int) : DiscreteHOGCell =
    new DiscreteHOGCell(grads.length, numParts, grads.map(g => (g * numParts).toInt))

}

/* A helper class to represent a discrete version of a HOG descriptor cell.
 * numBins: the number of gradient bins for this HOG cell
 * numPartitions: the number of bins in each gradient bin. The total number
 * of possible unique DiscreteHOGCells will be [num partitions]^[num gradient bins]
 */
class DiscreteHOGCell (val numBins: Int, val numPartitions: Int, val gradients: Array[Int])
  extends Serializable {

  require(gradients.length == numBins && gradients.forall(_ < numPartitions),
          "Gradient array does not match given parameters")

  def this (numBins: Int, numPartitions: Int) = this(numBins, numPartitions, new Array[Int](numBins))


  override def equals (other: Any) : Boolean = other match {

    case hogObj: DiscreteHOGCell => numBins == hogObj.numBins &&
                                    numPartitions == hogObj.numPartitions &&
                                    gradients.sameElements(hogObj.gradients)
    case _ => false

  }


  // piggyback off String hashing
  override def hashCode : Int = toString.hashCode


  def setGradients (grads: Array[Int]) : Unit = {

    assume(grads.length == numBins && grads.forall(_ < numPartitions),
           "Gradient array does not match given parameters")
    grads.copyToArray(gradients)

  }


  override def toString : String = {
    val minStr = numBins.toString + "-" + numPartitions + "-"
    minStr + gradients.map(_.toString).reduce((g1, g2) => g1 + "," + g2)
  }


  // a more verbose string representation of the cell
  def toStringVerbose : String = {
    val grads = "Array(" + gradients.map(_.toString).reduce((g1, g2) => g1 + ", " + g2) + ")"
    "DiscreteHOGCell: " + numBins + " bins, " + numPartitions + " bin partitions, gradients = " + grads
  }

}