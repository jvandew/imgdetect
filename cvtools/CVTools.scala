package imgdetect.cvtools

import imgdetect.util.{BoundingBox, Point}
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel, WindowConstants}
import org.opencv.core.{Core, CvType, Mat, MatOfByte, MatOfFloat, MatOfPoint,
                        Rect, Size, Scalar}
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import scala.math.{cos, Pi, pow, sin, toRadians}

// a static object containing useful tools for interfacing with OpenCV
// the goal is to have all OpenCV interaction happen within the cvtools package
object CVTools {

  // avoid namespace clashing
  type CVPoint = org.opencv.core.Point

  // dynamic OpenCV library must be loaded prior to calling any other methods
  // use CVTools.loadLibrary to accomplish this
  var libraryLoaded = false


  /* Given an array of HOG descriptors and the parameters that generated them,
   * aggregates the descriptors in the given bounding box by cell, averaging across
   * all blocks.
   */
  def aggregateHOGInBox (img: Mat) (box: BoundingBox) (descVals: Array[Float])
                        (winSize: Size, winStride: Size, blockSize: Size,
                         blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Float]]] = {

    val imgSize = img.size

    // hopefully self-explanatory
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt

    val cellsInBlockX = (blockSize.width / cellSize.width).toInt
    val cellsInBlockY = (blockSize.height / cellSize.height).toInt

    val cellsInWinX = (winSize.width / cellSize.width).toInt
    val cellsInWinY = (winSize.height / cellSize.height).toInt

    val blocksInWinX = (winSize.width/blockStride.width - (blockSize.width-blockStride.width)
                                                          / blockStride.width).toInt
    val blocksInWinY = (winSize.height/blockStride.height - (blockSize.height-blockStride.height)
                                                            / blockStride.height).toInt

    // unlike blockStride, winStride can be 0
    val winsInXDir = winStride.width match {
      case 0 => 1
      case w => (imgSize.width/w - (winSize.width - w)/w).toInt
    }
    val winsInYDir = winStride.height match {
      case 0 => 1
      case h => (imgSize.height/h - (winSize.height - h)/h).toInt
    }

    require(box.bottomRight.x <= cellsInXDir && box.bottomRight.y <= cellsInYDir,
            "Box does not fit in image; maybe you forgot to use cells instead of pixels?")

    val gradientStrengths = Array.ofDim[Float](cellsInYDir, cellsInXDir, numBins)
    val cellUpdateCounter = Array.ofDim[Int](cellsInYDir, cellsInXDir)
    var descDataIdx = 0

    // sum gradient strengths for each cell
    // TODO(jacob) if we're clever this can probably be done with a tabulate or similar
    for (winy <- 0 until winsInYDir) {
      for (winx <- 0 until winsInXDir) {
        for (blockx <- 0 until blocksInWinX) {
          for (blocky <- 0 until blocksInWinY) {
            for (celly <- 0 until cellsInBlockY) {
              for (cellx <- 0 until cellsInBlockX) {

                val (indx, indy) = (winx*cellsInWinX + blockx*cellsInBlockX + cellx,
                                    winy*cellsInWinY + blocky*cellsInBlockY + celly)

                // seven nested loops! a new record!
                for (bin <- 0 until numBins) {
                  gradientStrengths(indy)(indx)(bin) += descVals(descDataIdx)
                  descDataIdx += 1
                }

                cellUpdateCounter(indy)(indx) += 1

              }
            }
          }
        }
      }
    }

    // average gradient strengths
    for (celly <- 0 until cellsInYDir) {
      for (cellx <- 0 until cellsInXDir) {

        val numUpdates = cellUpdateCounter(celly)(cellx)

        for (bin <- 0 until numBins) {
          gradientStrengths(celly)(cellx)(bin) /= numUpdates
        }

      }
    }

    // TODO(jacob) can certainly be done more efficiently
    gradientStrengths.slice(box.topLeft.y, box.bottomRight.y).map(_.slice(box.topLeft.x, box.bottomRight.x))

  }


  /* Given an array of HOG descriptors and the parameters that generated them,
   * aggregates the descriptors by cell, averaging across all blocks.
   */
  def aggregateHOGInFullImage (img: Mat) (descVals: Array[Float])
                              (winSize: Size, winStride: Size, blockSize: Size,
                               blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Float]]] = {

    val allCellsBox = cellBox(img)(cellSize)
    aggregateHOGInBox(img)(allCellsBox)(descVals)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

  }


  /* Given an array of HOG descriptors and the parameters that generated them,
   * aggregates the descriptors by cell in each detection window, averaging across
   * all blocks.
   */
  def aggregateHOGWindows (img: Mat) (descVals: Array[Float])
                           (winSize: Size, winStride: Size, blockSize: Size,
                            blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Array[Float]]]] = {

    val imgSize = img.size

    // hopefully self-explanatory
    val cellsInBlockX = (blockSize.width / cellSize.width).toInt
    val cellsInBlockY = (blockSize.height / cellSize.height).toInt

    val cellsInWinX = (winSize.width / cellSize.width).toInt
    val cellsInWinY = (winSize.height / cellSize.height).toInt

    val blocksInWinX = (winSize.width/blockStride.width - (blockSize.width-blockStride.width)
                                                          / blockStride.width).toInt
    val blocksInWinY = (winSize.height/blockStride.height - (blockSize.height-blockStride.height)
                                                            / blockStride.height).toInt

    // unlike blockStride, winStride can be 0
    val winsInXDir = winStride.width match {
      case 0 => 1
      case w => (imgSize.width/w - (winSize.width - w)/w).toInt
    }
    val winsInYDir = winStride.height match {
      case 0 => 1
      case h => (imgSize.height/h - (winSize.height - h)/h).toInt
    }

    val gradientStrengths = Array.ofDim[Float](winsInXDir*winsInYDir, cellsInWinY, cellsInWinX, numBins)
    val cellUpdateCounter = Array.ofDim[Int](winsInXDir*winsInYDir, cellsInWinY, cellsInWinX)
    var descDataIdx = 0

    // sum gradient strengths for each cell
    // TODO(jacob) if we're clever this can probably be done with a tabulate or similar
    for (winy <- 0 until winsInYDir) {
      for (winx <- 0 until winsInXDir) {

        val windex = winy*winsInXDir + winx

        for (blockx <- 0 until blocksInWinX) {
          for (blocky <- 0 until blocksInWinY) {
            for (celly <- 0 until cellsInBlockY) {
              for (cellx <- 0 until cellsInBlockX) {

                val (indx, indy) = (blockx*cellsInBlockX + cellx, blocky*cellsInBlockY + celly)

                // seven nested loops! a new record!
                for (bin <- 0 until numBins) {
                  gradientStrengths(windex)(indy)(indx)(bin) += descVals(descDataIdx)
                  descDataIdx += 1
                }

                cellUpdateCounter(windex)(indy)(indx) += 1

              }
            }
          }
        }
      }
    }

    // average gradient strengths
    for (windex <- 0 until winsInXDir*winsInYDir) {
      for (celly <- 0 until cellsInWinY) {
        for (cellx <- 0 until cellsInWinX) {

          val numUpdates = cellUpdateCounter(windex)(celly)(cellx)

          for (bin <- 0 until numBins) {
            gradientStrengths(windex)(celly)(cellx)(bin) /= numUpdates
          }

        }
      }
    }

    gradientStrengths

  }


  // get a bounding box representing the size of the image in HOG cells
  def cellBox (img: Mat) (cellSize: Size) : BoundingBox = {

    val imgSize = img.size
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt

    BoundingBox(Point(0, 0), Point(cellsInXDir, cellsInYDir))

  }


  // compute a set of HOG descriptors for a given image and bounding box and visualize them via Swing
  def computeAndDisplayHOGBox (img: Mat) (box: BoundingBox) (winSize: Size, winStride: Size,
                                                             blockSize: Size, blockStride: Size,
                                                             cellSize: Size, numBins: Int)
                              (visScalar: Float) (windowTitle: String) : Unit = {

    val gradStrengths = computeHOGInBox(img)(box)(winSize, winStride, blockSize, blockStride, cellSize, numBins)
    val hogimg = getHOGBoxVisual(img)(box)(gradStrengths)(cellSize, numBins)(visScalar)

    displayImage(hogimg)(windowTitle)

  }


  // compute a set of HOG descriptors for a given image and visualize them via Swing
  def computeAndDisplayHOGFull (img: Mat) (winSize: Size, winStride: Size,
                                           blockSize: Size, blockStride: Size,
                                           cellSize: Size, numBins: Int)
                              (visScalar: Float) (windowTitle: String) : Unit = {

    val allCellsBox = cellBox(img)(cellSize)
    computeAndDisplayHOGBox(img)(allCellsBox)(winSize, winStride, blockSize, blockStride, cellSize, numBins)(visScalar)(windowTitle)

  }


  /* Compute the per-cell HOG descriptors for the given image in the given bounding box.
   * If there are overlapping blocks (i.e. blockSize != blockStride) descriptor values
   * for each cell will be averaged across all blocks it is contained in.
   *
   * Note: the given box should be in image cells as its units, not pixels
   */
  def computeHOGInBox (img: Mat) (box: BoundingBox) (winSize: Size, winStride: Size, blockSize: Size,
                                                     blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Float]]] = {

    val hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    val descVals = new MatOfFloat
    val locs = new MatOfPoint
    hog.compute(img, descVals, winStride, new Size(0, 0), locs)

    aggregateHOGInBox(img)(box)(descVals.toArray)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

  }


  /* Compute the per-cell HOG descriptors for the given image. If there are overlapping
   * blocks (i.e. blockSize != blockStride) descriptor values for each cell will be
   * averaged across all blocks it is contained in.
   */
  def computeHOGInFullImage (img: Mat) (winSize: Size, winStride: Size, blockSize: Size,
                                        blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Float]]] = {

    val allCellsBox = cellBox(img)(cellSize)
    computeHOGInBox(img)(allCellsBox)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

  }


  /* Compute the per-cell HOG descriptors for the given image over multiple windows.
   * If there are overlapping blocks (i.e. blockSize != blockStride) descriptor values
   * for each cell will be averaged across all blocks it is contained in.
   */
  def computeHOGWindows (img: Mat) (winSize: Size, winStride: Size, blockSize: Size,
                                    blockStride: Size, cellSize: Size, numBins: Int)
      : Array[Array[Array[Array[Float]]]] = {

    val hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    val descVals = new MatOfFloat
    val locs = new MatOfPoint
    hog.compute(img, descVals, winStride, new Size(0, 0), locs)

    aggregateHOGWindows(img)(descVals.toArray)(winSize, winStride, blockSize, blockStride, cellSize, numBins)

  }


  // crop an image at the specified bounding box
  def cropImage (img: Mat, box: BoundingBox) : Mat = {

    val rect = new Rect(box.topLeft.x, box.topLeft.y, box.width, box.height)
    new Mat(img, rect)

  }


  def displayBuffImage (bufImg: BufferedImage) (windowTitle: String) : Unit = {

    val frame = new JFrame(windowTitle)
    val label = new JLabel
    val icon = new ImageIcon

    icon.setImage(bufImg)
    label.setIcon(icon)

    frame.getContentPane.add(label)
    frame.setResizable(false)
    frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE)
    frame.pack
    frame.setVisible(true)

  }


  // display an image encoded as a Mat
  def displayImage (image: Mat) (windowTitle: String) : Unit = {

    val byteMat = new MatOfByte
    Highgui.imencode(".png", image, byteMat)

    val bytes = byteMat.toArray
    val in = new ByteArrayInputStream(bytes)
    val bufImg = ImageIO.read(in)

    displayBuffImage(bufImg)(windowTitle)

  }


  // Assistance with visualization algorithm found here:
  // http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
  def getHOGBoxVisual (image: Mat) (box: BoundingBox) (gradStrengths: Array[Array[Array[Float]]])
                      (cellSize: Size, numBins: Int) (visScalar: Float) : Mat = {


    val img = new Mat
    Imgproc.cvtColor(image, img, Imgproc.COLOR_GRAY2RGB)

    val binRads = toRadians(180.0 / numBins)

    for (celly <- box.topLeft.y until box.bottomRight.y) {
      for (cellx <- box.topLeft.x until box.bottomRight.x) {

        val tlX = cellx * cellSize.width
        val tlY = celly * cellSize.height
        val topLeft = new CVPoint(tlX, tlY)
        val botRight = new CVPoint(tlX + cellSize.width, tlY + cellSize.height)

        Core.rectangle(img, topLeft, botRight, new Scalar(100, 100, 100), 1)

        for (bin <- 0 until numBins) {

          val gradStr = gradStrengths(celly - box.topLeft.y)(cellx - box.topLeft.x)(bin)
          val currRad = bin*binRads + binRads / 2
          val dirVecX = cos(currRad)
          val dirVecY = sin(currRad)
          val maxVecLen = cellSize.width / 2
          val mx = tlX + cellSize.width / 2
          val my = tlY + cellSize.height / 2
          val xLen = dirVecX * gradStr * maxVecLen * visScalar
          val yLen = dirVecY * gradStr * maxVecLen * visScalar

          val p1 = new CVPoint(mx - xLen, my - yLen)
          val p2 = new CVPoint(mx + xLen, my + yLen)

          Core.line(img, p1, p2, new Scalar(0, 255, 0), 1)

        }
      }
    }

    img
  }


  def getHOGFullVisual (image: Mat) (gradStrengths: Array[Array[Array[Float]]])
                       (cellSize: Size, numBins: Int) (visScalar: Float) : Mat = {

    val allCellsBox = cellBox(image)(cellSize)
    getHOGBoxVisual(image)(allCellsBox)(gradStrengths)(cellSize, numBins)(visScalar)

  }


  // Used to visualize a trained model
  def getModelVisual (image: Mat) (gradStrengths: Array[Array[Array[Float]]])
                     (cellSize: Size, numBins: Int) (visScalar: Float) (visExpScalar: Float)
      : Mat = {

    val img = image.clone

    val box = cellBox(image)(cellSize)
    val binRads = toRadians(180.0 / numBins)

    for (celly <- box.topLeft.y until box.bottomRight.y) {
      for (cellx <- box.topLeft.x until box.bottomRight.x) {

        val tlX = cellx * cellSize.width
        val tlY = celly * cellSize.height

        for (bin <- 0 until numBins) {

          val gradStr = gradStrengths(celly - box.topLeft.y)(cellx - box.topLeft.x)(bin)
          val currRad = bin*binRads + (binRads + Pi) / 2
          val dirVecX = cos(currRad)
          val dirVecY = sin(currRad)
          val maxVecLen = cellSize.width / 2
          val mx = tlX + cellSize.width / 2
          val my = tlY + cellSize.height / 2
          val xLen = dirVecX * maxVecLen
          val yLen = dirVecY * maxVecLen

          val p1 = new CVPoint(mx - xLen, my - yLen)
          val p2 = new CVPoint(mx + xLen, my + yLen)
          val color = new Scalar(pow(gradStr * visScalar, visExpScalar) * 255)

          Core.line(img, p1, p2, color, 1)

        }
      }
    }

    img
  }


  // read in an image as grayscale
  def imreadGrayscale (imagePath: String) : Mat =
    Highgui.imread(imagePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE)


  // read in an image as greyscale (see what I did there? english is hard)
  def imreadGreyscale (imagePath: String) : Mat = imreadGrayscale(imagePath)


  // dynamic OpenCV library must be loaded prior to calling any other methods
  // it should also only be loaded once, which is why we synchronize here
  def loadLibrary () : Unit = this.synchronized {
    if (!libraryLoaded) {

      // TODO(jacob) this may not work
      //   if not use -Djava.library.path=share/OpenCV/java or similar at runtime
      // IDEA: re-compile OpenCV libs with -Wl --export-dynamic options found
      //   via random StackOverflow post
      // System.setProperty("java.library.path", "share/OpenCV/java")

      System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
      libraryLoaded = true
    }
  }


  // create a new OpenCV Size object and return it
  // this is meant more for use outside this package
  def makeSize (width: Double, height: Double) : Size = new Size(width, height)


  // create a new Mat representing a solid grayscale image
  def newGrayscaleImage (width: Int, height: Int) (color: Byte) : Mat =
    new Mat(width, height, CvType.CV_8U, new Scalar(color.toDouble))


  // create a new Mat representing a solid greyscale image (yes again...)
  def newGreyscaleImage (width: Int, height: Int) (color: Byte) : Mat =
    newGrayscaleImage(width, height)(color)

}
