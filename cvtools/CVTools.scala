package imgdetect.cvtools

import imgdetect.util.{BoundingBox, Point}
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel, WindowConstants}
import org.opencv.core.{Core, Mat, MatOfByte, MatOfFloat, MatOfPoint, Size, Scalar}
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import scala.math.{cos, sin, toRadians}

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
   * all blocks. Window sizes other than the image dimensions are currently ignored.
   *
   * Note: the given box should be in image cells as its units, not pixels
   *
   * TODO(jacob) implement multiple detection windows
   */
  def aggregateHOGInBox (img: Mat) (box: BoundingBox) (descVals: Array[Float])
                                                    (windowSize: Size, blockSize: Size,
                                                     blockStride: Size, cellSize: Size,
                                                     numBins: Int) : Array[Array[Array[Float]]] = {

    val winSize = img.size    // TODO(jacob) delete this line once multiple windows are implemented

    // hopefully self-explanatory
    val cellsInXDir = (winSize.width / cellSize.width).toInt
    val cellsInYDir = (winSize.height / cellSize.height).toInt
    val cellsInBlockX = (blockSize.width / cellSize.width).toInt
    val cellsInBlockY = (blockSize.height / cellSize.height).toInt
    val blocksInXDir = (winSize.width/blockStride.width - (blockSize.width-blockStride.width)
                          / blockStride.width).toInt
    val blocksInYDir = (winSize.height/blockStride.height - (blockSize.height-blockStride.height)
                          / blockStride.height).toInt

    if (box.bottomRight.x > cellsInXDir || box.bottomRight.y > cellsInYDir) {
      throw new IllegalArgumentException
          ("Box does not fit in image; maybe you forgot to use cells instead of pixels?")
    }

    val gradientStrengths = Array.ofDim[Float](cellsInYDir, cellsInXDir, numBins)
    val cellUpdateCounter = Array.ofDim[Int](cellsInYDir, cellsInXDir)
    var descDataIdx = 0

    // sum gradient strengths for each cell
    // TODO(jacob) if we're clever this can probably be done with a tabulate or similar
    for (blockx <- 0 until blocksInXDir) {
      for (blocky <- 0 until blocksInYDir) {
        for(celly <- 0 until cellsInBlockY) {
          for (cellx <- 0 until cellsInBlockX) {

            val (prevx, prevy) = ((blockx*blockStride.width / cellSize.width).toInt,
                                  (blocky*blockStride.height / cellSize.height).toInt)
            val (indx, indy) = (prevx + cellx, prevy + celly)

            for (bin <- 0 until numBins) {
              gradientStrengths(indy)(indx)(bin) += descVals(descDataIdx)
              descDataIdx += 1
            }

            cellUpdateCounter(indy)(indx) += 1

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
   * aggregates the descriptors by cell, averaging across all blocks. Window
   * sizes other than the image dimensions are currently ignored.
   *
   * TODO(jacob) implement multiple detection windows
   */
  def aggregateHOGInFullImage (img: Mat) (descVals: Array[Float])
                                         (winSize: Size, blockSize: Size, blockStride: Size,
                                          cellSize: Size, numBins: Int) : Array[Array[Array[Float]]] = {

    val imgSize = img.size
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt
    val allCellsBox = BoundingBox(Point(0, 0), Point(cellsInXDir, cellsInYDir))

    aggregateHOGInBox(img)(allCellsBox)(descVals)(winSize, blockSize, blockStride, cellSize, numBins)

  }


  // compute a set of HOG descriptors for a given image and bounding box and visualize them via Swing
  def computeAndDisplayHOGBox (img: Mat) (box: BoundingBox) (winSize: Size, blockSize: Size,
                                                             blockStride: Size, cellSize: Size,
                                                             numBins: Int)
                              (visScaler: Float) (windowTitle: String) : Unit = {

    val colorimg = new Mat
    Imgproc.cvtColor(img, colorimg, Imgproc.COLOR_GRAY2RGB)

    val gradStrengths = computeHOGInBox(img)(box)(winSize, blockSize, blockStride, cellSize, numBins)
    val hogimg = getHOGBoxVisual(colorimg)(box)(gradStrengths)(cellSize, numBins)(visScaler)

    val byteMat = new MatOfByte
    Highgui.imencode(".png", hogimg, byteMat)

    val bytes = byteMat.toArray
    val in = new ByteArrayInputStream(bytes)
    val bufImg = ImageIO.read(in)

    displayBuffImage(bufImg)(windowTitle)

  }


  // compute a set of HOG descriptors for a given image and visualize them via Swing
  def computeAndDisplayHOGFull (img: Mat) (winSize: Size, blockSize: Size,
                                          blockStride: Size, cellSize: Size,
                                          numBins: Int)
                              (visScaler: Float) (windowTitle: String) : Unit = {

    val imgSize = img.size
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt
    val allCellsBox = BoundingBox(Point(0, 0), Point(cellsInXDir, cellsInYDir))

    computeAndDisplayHOGBox(img)(allCellsBox)(winSize, blockSize, blockStride, cellSize, numBins)(visScaler)(windowTitle)

  }


  /* Compute the per-cell HOG descriptors for the given image in the given bounding box.
   * If there are overlapping blocks (i.e. blockSize != blockStride) descriptor values
   * for each cell will be averaged across all blocks it is contained in.
   *
   * Note 1: the given box should be in image cells as its units, not pixels
   * Note 2: winSize is currently ignored as window sizes other than the image dimensions
   *   are not currently supported
   *
   * TODO(jacob) implement multiple detection windows
   */
  def computeHOGInBox (img: Mat) (box: BoundingBox) (windowSize: Size, blockSize: Size,
                                                     blockStride: Size, cellSize: Size,
                                                     numBins: Int) : Array[Array[Array[Float]]] = {

    val winSize = img.size    // TODO(jacob) delete this line once multiple windows are implemented

    val hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    val descVals = new MatOfFloat
    val locs = new MatOfPoint
    hog.compute(img, descVals, new Size(0, 0), new Size(0, 0), locs)

    aggregateHOGInBox(img)(box)(descVals.toArray)(winSize, blockSize, blockStride, cellSize, numBins)

  }


  /* Compute the per-cell HOG descriptors for the given image. If there are overlapping
   * blocks (i.e. blockSize != blockStride) descriptor values for each cell will be
   * averaged across all blocks it is contained in.
   *
   * Note: winSize is currently ignored as window sizes other than the image dimensions
   *   are not currently supported
   *
   * TODO(jacob) implement multiple detection windows
   */
  def computeHOGInFullImage (img: Mat) (winSize: Size, blockSize: Size,
                                        blockStride: Size, cellSize: Size,
                                        numBins: Int) : Array[Array[Array[Float]]] = {

    val imgSize = img.size
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt
    val allCellsBox = BoundingBox(Point(0, 0), Point(cellsInXDir, cellsInYDir))

    computeHOGInBox(img)(allCellsBox)(winSize, blockSize, blockStride, cellSize, numBins)

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


  // Assistance with visualization algorithm found here:
  // http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
  def getHOGBoxVisual (image: Mat) (box: BoundingBox) (gradStrengths: Array[Array[Array[Float]]])
                      (cellSize: Size, numBins: Int) (visScaler: Float) : Mat = {

    val img = image.clone
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

          if (gradStr > 0) {

            val currRad = bin*binRads + binRads / 2
            val dirVecX = cos(currRad)
            val dirVecY = sin(currRad)
            val maxVecLen = cellSize.width / 2
            val mx = tlX + cellSize.width / 2
            val my = tlY + cellSize.height / 2
            val xLen = dirVecX * gradStr * maxVecLen * visScaler
            val yLen = dirVecY * gradStr * maxVecLen * visScaler

            val p1 = new CVPoint(mx - xLen, my - yLen)
            val p2 = new CVPoint(mx + xLen, my + yLen)

            Core.line(img, p1, p2, new Scalar(0, 255, 0), 1)

          }
        }

      }
    }

    img
  }


  def getHOGFullVisual (image: Mat) (gradStrengths: Array[Array[Array[Float]]])
                       (cellSize: Size, numBins: Int) (visScaler: Float) : Mat = {

    val imgSize = image.size
    val cellsInXDir = (imgSize.width / cellSize.width).toInt
    val cellsInYDir = (imgSize.height / cellSize.height).toInt
    val allCellsBox = BoundingBox(Point(0, 0), Point(cellsInXDir, cellsInYDir))

    getHOGBoxVisual(image)(allCellsBox)(gradStrengths)(cellSize, numBins)(visScaler)

  }


  // read in an image as grayscale
  def imreadGrayscale (imagePath: String) : Mat =
    Highgui.imread(imagePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE)


  // read in an image as greyscale (see what I did there? english is hard)
  def imreadGreyscale (imagePath: String) : Mat = imreadGrayscale(imagePath)


  // dynamic OpenCV library must be loaded prior to calling any other methods
  // it should also only be loaded once, which is why we synchronize here
  def loadLibrary : Unit = this.synchronized {
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

}