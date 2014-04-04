package imgdetect.cvtools

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel, WindowConstants}
import org.opencv.core.{Core, Mat, MatOfByte, MatOfFloat, MatOfPoint, Point, Size, Scalar}
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import scala.math.{cos, sin, toRadians}

// a static object containing useful tools for interfacing with OpenCV
object CVTools {

  // dynamic OpenCV library must be loaded prior to calling any other methods
  // use CVTools.loadLibrary to accomplish this
  var libraryLoaded = false

  // compute a set of HOG descriptors for a given image and visualize them via Swing
  def computeAndDisplayHOG (img: Mat) (winSize: Size, blockSize: Size,
                                       blockStride: Size, cellSize: Size,
                                       numBins: Int) (visScaler: Float) (windowTitle: String) : Unit = {

    val colorimg = new Mat
    Imgproc.cvtColor(img, colorimg, Imgproc.COLOR_GRAY2RGB)

    val hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    val descVals = new MatOfFloat
    val locs = new MatOfPoint
    hog.compute(img, descVals, new Size(0, 0), new Size(0, 0), locs)

    val hogimg = getHOGDescriptorVisual(colorimg)(descVals.toArray)(winSize,
                                        blockSize, blockStride, cellSize, numBins)(visScaler)
    val byteMat = new MatOfByte
    Highgui.imencode(".png", hogimg, byteMat)

    val bytes = byteMat.toArray
    val in = new ByteArrayInputStream(bytes)
    val bufImg = ImageIO.read(in)

    displayBuffImage(bufImg)(windowTitle)

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
  def getHOGDescriptorVisual (image: Mat) (descVals: Array[Float])
                             (winSize: Size, blockSize: Size,
                              blockStride: Size, cellSize: Size, numBins: Int)
                             (visScaler: Float) : Mat = {

    val img = image.clone
    val binRads = toRadians(180.0 / numBins)
    val cellsInXDir = (winSize.width / cellSize.width).toInt
    val cellsInYDir = (winSize.height / cellSize.height).toInt
    val cellsInBlockX = (blockSize.width / cellSize.width).toInt
    val cellsInBlockY = (blockSize.height / cellSize.height).toInt

    val gradientStrengths = Array.ofDim[Float](cellsInYDir, cellsInXDir, numBins)
    val cellUpdateCounter = Array.ofDim[Int](cellsInYDir, cellsInXDir)

    val blocksInXDir = (winSize.width/blockStride.width - (blockSize.width-blockStride.width) / blockStride.width).toInt
    val blocksInYDir = (winSize.height/blockStride.height - (blockSize.height-blockStride.height) / blockStride.height).toInt
    var descDataIdx = 0

    // sum gradient strengths for each cell
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

    for (celly <- 0 until cellsInYDir) {
      for (cellx <- 0 until cellsInXDir) {

        val tlX = cellx * cellSize.width
        val tlY = celly * cellSize.height
        val topLeft = new Point(tlX, tlY)
        val botRight = new Point(tlX + cellSize.width, tlY + cellSize.height)

        Core.rectangle(img, topLeft, botRight, new Scalar(100, 100, 100), 1)

        for (bin <- 0 until numBins) {

          val gradStrength = gradientStrengths(celly)(cellx)(bin)

          if (gradStrength > 0) {

            val currRad = bin*binRads + binRads/2
            val dirVecX = cos(currRad)
            val dirVecY = sin(currRad)
            val maxVecLen = cellSize.width / 2
            val mx = tlX + cellSize.width/2
            val my = tlY + cellSize.height/2
            val xLen = dirVecX*gradStrength*maxVecLen*visScaler
            val yLen = dirVecY*gradStrength*maxVecLen*visScaler

            val p1 = new Point(mx - xLen, my - yLen)
            val p2 = new Point(mx + xLen, my + yLen)

            Core.line(img, p1, p2, new Scalar(0, 255, 0), 1)

          }
        }

      }
    }

    img
  }


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

}