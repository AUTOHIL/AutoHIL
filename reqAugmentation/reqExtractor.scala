import io.shiftleft.semanticcpg.language._
import java.io.{BufferedWriter, FileWriter, PrintWriter}
import io.shiftleft.codepropertygraph.generated.Cpg
import io.shiftleft.codepropertygraph.cpgloading.{CpgLoader, ProtoCpgLoader}
import scala.collection.mutable.{ArrayBuffer, Map => MutableMap, Set => MutableSet}
import scala.io.Source

val moduleName = "<your_module_name>"  // User Define

object CallGraphGenerator {

  // User Define
  var cpg: Cpg = CpgLoader.load(s"./cpg/$moduleName/cpg_overlay.bin")
  val functionInfoMap = MutableMap[String, (String, Int, Int)]()  // (filename, lineStart, lineEnd)

  /**
   * Read all method nodes appearing in the specified .dot file
   * Obtain all method nodes by parsing both endpoints of edges and deduplicating them
   */
  def readMethodsFromDot(dotFilePath: String): Set[String] = {
    val methodNodes = MutableSet[String]()
    
    try {
      val source = Source.fromFile(dotFilePath, "UTF-8")
      val lines = source.getLines()
      
      // Regex to match edge format in DOT files:  "node1" -> "node2";
      val edgePattern = """"([^"]+)"\s*->\s*"([^"]+)"""".r
      
      lines.foreach { line =>
        edgePattern.findAllMatchIn(line).foreach { matchResult =>
          // Extract source node and target node
          val sourceNode = matchResult.group(1)
          val targetNode = matchResult.group(2)
          
          // Add to set (auto-deduplicate)
          methodNodes.add(sourceNode)
          methodNodes.add(targetNode)
        }
      }
      
      source.close()
      println(s"Successfully read ${methodNodes.size} unique method nodes from $dotFilePath")
      
    } catch {
      case e: Exception =>
        println(s"Error reading DOT file $dotFilePath: ${e.getMessage}")
        e.printStackTrace()
    }
    
    methodNodes.toSet
  }

  // Recursive function
  def exploreCalls(
    method: String,
    callGraph: MutableMap[String, MutableSet[String]]
  ): Unit = {
    if (callGraph.contains(method)) return 

    val methodNode = cpg.method.name(method).headOption
    methodNode match {
      case Some(m: Method) => 
        val lineStart = m.lineNumber.getOrElse(-1)
        val lineEnd = m.lineNumberEnd.getOrElse(-1)
        val fileName = m.file.name.l.headOption.getOrElse("unknown_file")

        if (lineStart != lineEnd && !fileName.endsWith(".h")) {
          functionInfoMap.get(method) match {
          // If line number info has already been stored, it indicates same-named functions exist and there is a conflict; report an error
            case Some((prevFile, prevStart, prevEnd)) => 
              if (prevFile != fileName || prevStart != lineStart || prevEnd != lineEnd) {
                println(
                  s"Inconsistent function line info for '$method': " +
                  s"previous=($prevFile:$prevStart, $prevEnd), current=($fileName:$lineStart, $lineEnd)"
                )
              }
              // else: info is consistent; no need to write again
            case None =>
              // If line number info has not been stored, add it to the function info map
              functionInfoMap(method) = (fileName, lineStart, lineEnd)
          }
        
          // Get all actual function calls of the current function (exclude operators, macro definitions, etc.)
          var callees = m.callee
            .filter(callee => {
                  callee.astParentType != "NAMESPACE_BLOCK" &&
                  callee.lineNumber.isDefined &&
                  callee.filename != "<empty>"
              }) // Exclude irrelevant functions like namespaces
            .name.filter(isReserved)
            .toSet
          
          // Also consider calling function pointers via struct variables (LOCAL variables, e.g., canmgr_MsgDataProcessTable)
          // First, get the local names invoked inside the function via closureBindingId — capture nodes
          val local_sink = m.local
            .filter(_.closureBindingId.isDefined)
            .toSet
          // Then, find the concrete implementation in source code corresponding to the capture node via local
          // Check whether the AST of the implementation contains function pointers
          local_sink.foreach { local => 
            val local_source = cpg.local.name(local.name).headOption
            local_source match {
              case Some(lo: Local) => 
                val impl = cpg.call.name("<operator>.assignment")
                          .where(_.argument(1).code.filter(_.contains(lo.name)))
                          .headOption
                impl match {
                  case Some(impl: Call) => 
                    val callPointerSet = impl.ast.isMethodRef.methodFullName.toSet
                    callees ++= callPointerSet
                  case None => 
                }
              case None => 
            }
          }

          // If the current function calls other functions, add them to the call graph
          if (callees.nonEmpty) {
            callGraph += (method -> MutableSet.from(callees))
            callees.foreach { callee =>
              // Recursively traverse each callee; depth + 1
              exploreCalls(
                callee,
                callGraph
              ) 
            }
          }
        }

      case None => 
        println(s"Warning: method '$method' not found in CPG.")
    }
  }

  def generateCallGraph(method: String): Unit = {

    var callGraph = MutableMap[String, MutableSet[String]]()
    exploreCalls(
      method,  
      callGraph
    )

    var dotGraph = "digraph G {\n"  // Define a directed graph
    // Generate DOT syntax for each function and its call relations
    callGraph.foreach { case (caller, callees) =>
      callees.foreach { callee =>
        dotGraph += s"""  "$caller" -> "$callee";\n"""
      }
    }

    // --- Write DOT file ---
    dotGraph += "}\n"  // End graph definition

    val fileName = s"./output/$moduleName/cg/${method}.dot"
    val writer = new BufferedWriter(new FileWriter(fileName))
    writer.write(dotGraph)
    writer.close()
    println(s"DOT graph has been written to $fileName")

  }

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
        // Fetch the range of lines (start to end, inclusive)
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

  def getMacros(methodName: String): String = {
    val methodNode = cpg.method(methodName).headOption
    methodNode match {
      case Some(method: Method) =>
        val macro_set = MutableSet[String]()
        method.call.foreach { call => 
          if (call.dispatchType == DispatchTypes.INLINED) {
            val macroBodyOpt = call.astChildren.isBlock.maxByOption(_.order).iterator.expressionDown.headOption
            val macroBody = macroBodyOpt match {
              case Some(expr) => expr.code
                .replaceAll("\\\\\\s*\\n", "")  // Remove backslash line-continuation (\\\n)
                .replaceAll("\\n", " ")  // Replace remaining newlines with spaces
                .replaceAll("\\s+", " ")  // Collapse multiple consecutive spaces into one
                .replaceAll("\\t", " ")
                .replaceAll("\\t+", " ")
                .replaceAll("\\r", " ")
                .trim  // Trim leading/trailing spaces
              case None => ""
            }
            val macroDef = s"#define ${call.code} ${macroBody}"
            if (!macro_set.contains(macroDef)) {
              macro_set.add(macroDef)
            }
          }
        }
        macro_set.mkString("; ")
      case None => ""
    }
  }

  def getCalleeNameOfMethod(methodName: String): String = {
    val callee_set = MutableSet[String]()
    this.cpg.method(methodName).headOption match {
      case Some(method) =>
        method.call.foreach { call => 
          if (call.dispatchType != DispatchTypes.INLINED) {
            val callee_name = call.methodFullName
            if (isReserved(callee_name)) {
              callee_set.add(callee_name)
            }
          }
        }
      case None =>
        println(s"Warning: Method '$methodName' not found in CPG")
    }
    callee_set.mkString(",")
  }

  
  /* ------------Core Functions------------ */
  def MethodCallGraph(methodName: String): Unit = {
    generateCallGraph(methodName)

    // --- Write functionInfoMap to JSON ---
    val jsonFile = s"./output/function_info_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      val fullPath = s"./projClosure/CanMgr/$file"
      val code = readSourceSnippet(fullPath, start, end)
      val macroText = getMacros(name)
        .replace("\\", "\\\\")  // Escape backslashes for JSON
        .replace("\"", "\\\"")  // Escape double quotes for JSON
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "is_atomic": ${name == methodName},
       |    "macro": "$macroText"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }
  
  def batchMethodCallGraph(): Unit = {
    // Configure module name
    val methodList = MutableSet("CanMgr_MainFunction", "CanMgr_Initialization")
    methodList.foreach { methodName => generateCallGraph(methodName) }
    
    // --- Write functionInfoMap to JSON ---
    val jsonFile = s"./output/$moduleName/summary/function_info_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      // User Define
      val fullPath = s"./projClosure/$moduleName/$file"
      val code = readSourceSnippet(fullPath, start, end)
      val macroText = getMacros(name)
      val calleeList = getCalleeNameOfMethod(name)
        .replace("\\", "\\\\")  // Escape backslashes for JSON
        .replace("\"", "\\\"")  // Escape double quotes for JSON
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "macro": "$macroText",
       |    "callee": "$calleeList"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }

  def batchMethodCallGraph2(): Unit = {
    // Configure module name
    val dotPath = MutableSet("./output/Sma7Drvr/cg/1/Sma7Drvr_Initialization.dot",
                             "./output/Sma7Drvr/cg/1/Sma7Drvr_MainFunction.dot")
    dotPath.foreach { dotPath => 
      val methodList = readMethodsFromDot(dotPath)
      methodList.foreach { methodName => generateCallGraph(methodName) }
    }
    
    // --- Write functionInfoMap to JSON ---
    val jsonFile = s"./output/$moduleName/summary/function_info_map2.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      // User Define
      val fullPath = s"./projClosure/$moduleName/Projects/$file"
      val code = readSourceSnippet(fullPath, start, end)
      val macroText = getMacros(name)
      val calleeList = getCalleeNameOfMethod(name)
        .replace("\\", "\\\\")  // Escape backslashes for JSON
        .replace("\"", "\\\"")  // Escape double quotes for JSON
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "macro": "$macroText",
       |    "callee": "$calleeList"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }


  def fileCallGraph(fileName: String): Unit = {
    val methods = cpg.file(fileName).method
                .name.filter(isReserved)
                .toSet
    methods.foreach { method => 
      generateCallGraph(method)
    }
  }

  def batchFileCallGraph(): Unit = {
    val files = cpg.file.name.toSet
    
    files.foreach { file => 
      fileCallGraph(file)
    }

    // --- Write functionInfoMap to JSON ---
    // User Define
    val jsonFile = s"./output/$moduleName/summary/function_info_map1.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      // User Define
      val fullPath = s"./projClosure/$moduleName/Platform/$file"
      val code = readSourceSnippet(fullPath, start, end)
      val macroText = getMacros(name)
      val calleeList = getCalleeNameOfMethod(name)
        .replace("\\", "\\\\")  // Escape backslashes for JSON
        .replace("\"", "\\\"")  // Escape double quotes for JSON
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "macro": "$macroText",
       |    "callee": "$calleeList"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }

}
