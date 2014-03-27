import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel}
import org.opencv.core.{Core, Mat, MatOfByte}
import org.opencv.highgui.Highgui

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

  def main (args: Array[String]) = {

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    val byteMat = new MatOfByte

    val img = Highgui.imread("resources/lena.png",
                             Highgui.CV_LOAD_IMAGE_GRAYSCALE)
    Highgui.imencode(".png", img, byteMat)
    
    val bytes = byteMat.toArray
    val in = new ByteArrayInputStream(bytes)
    val bufImg = ImageIO.read(in)

    displayBuffImage(bufImg)

  }
}
