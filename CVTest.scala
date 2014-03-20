import org.opencv.core.{Core, Mat, MatOfFloat}
import org.opencv.highgui.Highgui

object CVTest {

  def main (args: Array[String]) = {
    println("Hello")
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    println("locked and loaded: " + Core.VERSION)
    val img = Highgui.imread("resources/lena.png")
    println(img)
  }
}
