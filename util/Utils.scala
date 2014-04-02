import java.io.{File, FileInputStream}

// Some useful utilities
object Utils {

  def readFile (filename: String) : String =
    readFile(new File(filename))

  // read the contents of a file
  def readFile (file: File) : String = {
    val fileIn = new FileInputStream(file)
    val bytes = new Array[Byte](file.length.toInt)
    fileIn.read(bytes)
    fileIn.close
    new String(bytes)
  }

}