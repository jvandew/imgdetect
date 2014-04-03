package imgdetect.util

import java.io.{File, FileInputStream}

// Some useful utilities
object Utils {

  // join two file paths together
  def joinPath (path1: String, path2: String) : String = {

    if (path1.endsWith(File.separator)) {
      path1 + path2
    }
    else {
      path1 + File.separator + path2
    }
  }

  def readFile (filename: String) : String = readFile(new File(filename))

  // read the contents of a file
  def readFile (file: File) : String = {

    val fileIn = new FileInputStream(file)
    val bytes = new Array[Byte](file.length.toInt)
    fileIn.read(bytes)
    fileIn.close
    new String(bytes)
  }

}