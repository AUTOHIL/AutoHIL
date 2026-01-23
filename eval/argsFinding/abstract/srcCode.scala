import io.shiftleft.semanticcpg.language._
import java.io.{BufferedWriter, FileWriter, PrintWriter}
import java.io.{BufferedReader, FileReader}
import io.shiftleft.codepropertygraph.generated.Cpg
import io.shiftleft.codepropertygraph.cpgloading.{CpgLoader, ProtoCpgLoader}
import scala.collection.mutable.{ArrayBuffer, Map => MutableMap, Set => MutableSet}
import scala.io.Source

val moduleName = "<your_module_name>"

object srcCodeAnalyzer {
  var cpg: Cpg = CpgLoader.load(s"./reqAugmentation/cpg/$moduleName/cpg_overlay.bin")
  var functionInfoMap = MutableMap[String, (String, Int, Int)]()

  def isReserved(method: String): Boolean = {
    !method.contains("<metaClassAdapter>") &&
    !method.contains("<") &&
    !method.contains("__") &&
    !method.contains("unknown") &&
    !(method == "NULL" || method == "TRUE" || method == "FALSE") &&
    !(method.toUpperCase == method && method.nonEmpty)
  }

  def readSourceSnippet(filePath: String, start: Int, end: Int): String = {
    try {
      // Check whether the file exists
      val file = new java.io.File(filePath)
      if (!file.exists()) {
        println(s"File does not exist: $filePath")
        return ""
      }
      
      if (!file.canRead()) {
        println(s"File cannot be read: $filePath")
        return ""
      }
      
      // Try reading the file with multiple encodings
      val encodings = List("UTF-8", "ISO-8859-1", "GBK", "ASCII")
      var lines: List[String] = List.empty
      var successful = false
      
      for (encoding <- encodings if !successful) {
        try {
          val source = Source.fromFile(filePath, encoding)
          lines = source.getLines().toList
          source.close()
          successful = true
        } catch {
          case _: Exception => 
            // Try the next encoding
        }
      }
      
      if (!successful) {
        println(s"Failed to read file with any encoding: $filePath")
        return ""
      }
      
      if (lines.isEmpty) {
        println(s"File is empty: $filePath")
        return ""
      }
      
      // Scala line numbers start from 1, while List indices start from 0
      val result = if (start == end) {
        // Fetch a single line
        if (start > 0 && start <= lines.length) lines(start - 1) else ""
      } else {
        // Fetch the range of lines (from start to end, inclusive)
        lines.slice(start - 1, end).mkString("\n")
      }
      
      result
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    } catch {
      case e: Exception => 
        println(s"Failed to read source for $filePath:$start-$end: ${e.getMessage}")
        ""
    }
  }

  // Read the JSON file of the "file set"
  def readFilesFromJson(path: String): Set[String] = {
    val br = new BufferedReader(new FileReader(path))
    val sb = new StringBuilder
    var line = br.readLine()
    while (line != null) {
      sb.append(line)
      line = br.readLine()
    }
    br.close()
    // Remove [ ] and ", then split by comma
    val cleaned = sb.toString().replace("[", "").replace("]", "").replace("\"", "")
    val files = cleaned.split(",")
                        .map(_.trim)
                        .filter(_.nonEmpty)
                        .filter(_.endsWith(".c"))
                        .filter { fullPath =>
                          // 3. Extract the file name (the part after the last '/' in the path)
                          val lastSlashIndex = fullPath.lastIndexOf('/')
                          val fileName = if (lastSlashIndex >= 0) {
                            fullPath.substring(lastSlashIndex + 1)
                          } else {
                            fullPath // If there is no '/', the entire string is the file name
                          }
                          
                          // 4. Apply all filtering conditions
                          // Must end with .c AND (file name starts with "Can" OR starts with "Com")
                          fullPath.endsWith(".c") && (fileName.startsWith("Can") || fileName.startsWith("Com"))
                        }
                        .toSet
    files
  }

  def saveInfoToMap(methodName: String): Unit = {
    val methodNode = cpg.method.name(methodName).headOption
    methodNode match {
      case Some(m: Method) => 
        val lineStart = m.lineNumber.getOrElse(-1)
        val lineEnd = m.lineNumberEnd.getOrElse(-1)
        val fileName = m.file.name.l.headOption.getOrElse("unknown_file")
        
        if (lineStart != lineEnd && !fileName.endsWith(".h")) {
          functionInfoMap.get(methodName) match {
          // If line number info has already been stored, it indicates same-named functions exist and there is a conflict; report an error
            case Some((prevFile, prevStart, prevEnd)) => 
              if (prevFile != fileName || prevStart != lineStart || prevEnd != lineEnd) {
                println(
                  s"Inconsistent function line info for '$methodName': " +
                  s"previous=($prevFile:$prevStart, $prevEnd), current=($fileName:$lineStart, $lineEnd)"
                )
              }
              // else: info is consistent; no need to write again
            case None =>
              // If line number info has not been stored, add it to the function info map
              functionInfoMap(methodName) = (fileName, lineStart, lineEnd)
          }
        }
      case None => 
        println(s"Warning: method '$methodName' not found in CPG.")
    
    }
  }

  def main(): Unit = {
    val files = readFilesFromJson(s"./reqAugmentation/output/$moduleName/cTarFiles.json")
    files.foreach { file => 
      val methods = cpg.file(file).method.name.filter(isReserved).toSet
      methods.foreach { method => 
        saveInfoToMap(method)
      }
    }
    
    // --- Write functionInfoMap to JSON
    val jsonFile = s"./argsFinding/abstract/function_info_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      val fullPath = s"./reqAugmentation/projClosure/$moduleName/$file"
      val code = readSourceSnippet(fullPath, start, end)
        .replace("\\", "\\\\")  // Escape backslashes for JSON
        .replace("\"", "\\\"")  // Escape double quotes for JSON
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }
}
