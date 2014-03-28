import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel}
import org.opencv.core.{Core, Mat, MatOfByte, MatOfFloat, MatOfPoint, Point, Size, Scalar}
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import scala.math.{cos, sin, toRadians}

object CVTest {

  def displayBuffImage (bufImg: BufferedImage) : Unit = {
    
    val frame = new JFrame("Test Image")
    val label = new JLabel
    val icon = new ImageIcon

    icon.setImage(bufImg)
    label.setIcon(icon)

    frame.getContentPane.add(label)
    frame.setResizable(false)
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.pack
    frame.setVisible(true)

  }

  // Assistance with visualization algorithm found here:
  // http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
  def getHOGDescriptorVisual (image: Mat)
                             (descVals: Array[Float], winSize: Size,
                                 cellSize: Size, numBins: Int)
                             (visScaler: Float) : Mat = {

    val img = image.clone
    val binRads = toRadians(180.0 / numBins)
    val cellsInXDir = winSize.width / cellSize.width
    val cellsInYDir = winSize.height / cellSize.height
    val numCells = cellsInXDir * cellsInYDir

    val gradientStrengths = Array.ofDim[Float](cellsInYDir, cellsInXDir, numBins)
    val cellUpdateCounter = Array.ofDim[Int](cellsInYDir, cellsInXDir)

    val blocksInXDir = cellsInXDir - 1
    val blocksInYDir = cellsInYDir - 1
    var descDataIdx = 0

    // sum gradient strengths for each cell
    for (blockx <- 0 until blocksInXDir) {
      for (blocky <- 0 until blocksInYDir) {
        for (cellNum <- 0 until 4) {

          val (cellx, celly) = cellNum match {
            case 0 => (blockx, blocky)
            case 1 => (blockx, blocky + 1)
            case 2 => (blockx + 1, blocky)
            case 3 => (blockx + 1, blocky + 1)
          }

          for (bin <- 0 until numBins) {
            gradientStrengths(celly)(cellx)(bin) += descVals(descDataIdx)
            descDataIdx += 1
          }

          cellUpdateCounter(celly)(cellx) += 1

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
            val mx = drawX + cellSize.width/2
            val my = drawY + cellSize.height/2
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

  def main (args: Array[String]) = {

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    val byteMat = new MatOfByte

    val img = Highgui.imread("resources/lena.png",
                             Highgui.CV_LOAD_IMAGE_GRAYSCALE)
    val colorimg = new Mat
    Imgproc.cvtColor(img, colorimg, Imgproc.COLOR_GRAY2RGB)

    val winSize = new Size(512, 512) // size of lena
    val blockSize = new Size(64, 64)
    val blockStride = new Size(32, 32)
    val cellSize = new Size(32, 32)
    val numBins = 9

    val hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    val descVals = new MatOfFloat
    val locs = new MatOfPoint
    hog.compute(img, descVals, new Size(0, 0), new Size(0, 0), locs)

    val hogimg = getHOGDescriptorVisual(colorimg)(descVals.toArray)(winSize)(cellSize)(3.0f)
    Highgui.imencode(".png", hogimg, byteMat)
    
    val bytes = byteMat.toArray
    val in = new ByteArrayInputStream(bytes)
    val bufImg = ImageIO.read(in)

    displayBuffImage(bufImg)

  }
}
