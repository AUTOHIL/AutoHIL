import io.shiftleft.semanticcpg.language._
import java.io.{BufferedWriter, FileWriter, PrintWriter}
import java.io.{BufferedReader, FileReader}
import io.shiftleft.codepropertygraph.generated.Cpg
import io.shiftleft.codepropertygraph.cpgloading.{CpgLoader, ProtoCpgLoader}
import scala.collection.mutable.{ArrayBuffer, Map => MutableMap, Set => MutableSet}
import scala.io.Source

object CallGraphGenerator {
  val project_interface = "<your_hil_project_entry_name>"
  var cpg: Cpg = CpgLoader.load("./cpg/cpg.bin")
  val functionInfoMap = MutableMap[String, (String, Int, Int)]()  // (filename, lineStart, lineEnd)
  val functionClsMap = MutableMap[String, String]()  // (methodName, className)
  var classesToAnalyze = MutableSet[TypeDecl]()
  val attrClsMap = MutableMap[String, MutableMap[String, String]]()  // Class Name -> {Attribute Name -> Constructor Class}
  
  // Store all inheritance chains: each chain ordered from subclass to superclass
  val allInheritanceChains = MutableSet[ArrayBuffer[String]]()

  // Recursive function -- generate call graph for specified function
  // Maintained static analysis info: fileName, lineStart, lineEnd, className
  def exploreCalls(
    method: String,
    // depth: Int, 
    callGraph: MutableMap[String, MutableSet[String]]
  ): Unit = {
    // Set max recursion depth
    // val maxDepth = 5
    // If max depth reached, stop recursion
    // if (depth > maxDepth) return

    if (callGraph.contains(method)) return 

    val methodNode = cpg.method.name(method).headOption
    methodNode match {
      case Some(m: Method) => 
        val lineStart = m.lineNumber.getOrElse(-1)
        val lineEnd = m.lineNumberEnd.getOrElse(-1)
        val fileName = m.file.name.l.headOption.getOrElse("unknown_file")

        if (fileName == "unknown_file" && lineStart == -1 && lineEnd == -1) { }
        else {
          functionInfoMap.get(method) match {
            // If line number info is already stored, it implies a duplicate function name conflict; raise error
            case Some((prevFile, prevStart, prevEnd)) => 
              if (prevFile != fileName || prevStart != lineStart || prevEnd != lineEnd) {
                throw new RuntimeException(
                  s"Inconsistent function line info for '$method': " +
                  s"previous=($prevFile:$prevStart, $prevEnd), current=($fileName:$lineStart, $lineEnd)"
                )
              }
              // else info matches, no need to rewrite
            case None =>
              // If line number info not stored, add to function info map
              functionInfoMap(method) = (fileName, lineStart, lineEnd)
          }
        
          // Get all actual function calls of current function (exclude operators, macros, etc.)
          val callees = m.callee
            .filter(callee => {
                  callee.astParentType != "NAMESPACE_BLOCK"
              }) // Exclude unrelated functions like namespaces
            .name.filter(isReserved)
            .toSet

          // If current function calls other functions, add to call graph
          if (callees.nonEmpty) {
            callGraph += (method -> MutableSet.from(callees))
            callees.foreach { callee =>
              // Recursively traverse each called function, increment depth
              exploreCalls(
                callee, 
                // depth + 1, 
                callGraph
              )
            }
          }
        }

      case None => 
        println(s"Warning: method '$method' not found in CPG.")
    }
  }

  // Outer recursion call -- batch generate call graph and write to DOT
  def generateCallGraph(method: String) = {

    var callGraph = MutableMap[String, MutableSet[String]]()
    exploreCalls(
      method, 
      // 1, 
      callGraph
    )

    var dotGraph = "digraph G {\n"  // Define a digraph
    // Generate DOT syntax for each function and its call relations
    callGraph.foreach { case (caller, callees) =>
      callees.foreach { callee =>
        dotGraph += s"""  "$caller" -> "$callee";\n"""
      }
    }

    // --- Write DOT file ---
    dotGraph += "}\n"  // End graph definition
    val fileName = s"./callgraph/${method}.dot"
    val writer = new BufferedWriter(new FileWriter(fileName))
    writer.write(dotGraph)
    writer.close()
  }

  /******** Utils *********/
  // Determine if it is a reserved function
  def isReserved(method: String): Boolean = {
    !method.contains("<metaClassAdapter>") &&
    !method.contains("<") &&
    !method.contains("__") &&
    method != "replace" && method != "startswith" &&
    method != "append" && method != "print_help"
  }

  // Read source code snippet
  def readSourceSnippet(filePath: String, start: Int, end: Int): String = {
    try {
      val source = Source.fromFile(filePath)
      val lines = source.getLines().toList
      source.close()
      // Scala line numbers start at 1, List starts at 0
      // Extract lines start ~ end - 1
      lines.slice(start - 1, end - 1).mkString("\n")
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

  // Read the "file collection" JSON file
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
    val files = cleaned.split(",").map(_.trim).filter(_.nonEmpty).toSet
    files
  }

  // Get all attribute assignment operations for a *specific class* and store in attrClsMap
  def getAttrAssignments(currentCls: TypeDecl): Unit = {
    val initMethodOpt = currentCls.member.name("__init__").dynamicTypeHintFullName.flatMap(cpg.method.fullName(_)).headOption
    val currentClassName = currentCls.name

    // Ensure current class has corresponding inner map in the map
    if (!attrClsMap.contains(currentClassName)) {
      attrClsMap(currentClassName) = MutableMap[String, String]()
    }

    initMethodOpt match {
      case Some(initMethod) =>
        // println(initMethod.fullName)
        // Find all assignment operations
        initMethod.call.name("<operator>.assignment").foreach { assignment =>
          val lhs = assignment.argument(1).collectFirst {
            case callNode: nodes.Call => callNode
          } // LHS of assignment (self.attributeName)
          val rhs = assignment.argument(2).collectFirst {
            case callNode: nodes.Call => callNode
          } // RHS of assignment

          if (lhs!=None && rhs!=None) {
            // Check if LHS is member access in self.xxx form
            if (lhs.name("<operator>.fieldAccess").nonEmpty) {
              val fieldAccess = lhs.name("<operator>.fieldAccess").head
              val attributeName = fieldAccess.argument(2).code.mkString("")

              // Try to get creating class info from RHS
              var creatingClassName: String = ""

              if (rhs.cfgPrev.head.isInstanceOf[Identifier]) {
                creatingClassName = rhs.cfgPrev.head.code.mkString("")
              }

              // Add created class name and attribute name to current class map
              // attrClsMap(currentClassName)(creatingClassName) = attributeName
              // Add attribute name and created class name to current class map (Attribute Name -> Constructor Class)
              // attrClsMap(currentClassName)(attributeName) = creatingClassName
              // Use minimal subclass to replace original creating class
              val minimalCreatingCls = findMinimalSubclass(creatingClassName)
              attrClsMap(currentClassName)(attributeName) = minimalCreatingCls
            }
          }
        }
      case None =>
        println(s"Warning: __init__ method not found for class $currentClassName")
    }
  }
  
  // Find minimal subclass of a class (first subclass in inheritance chain)
  def findMinimalSubclass(className: String): String = {
    allInheritanceChains.find(_.contains(className)) match {
      case Some(chain) => chain.head // Return first element of chain (most specific subclass)
      case None => className // If inheritance chain not found, return original class name
    }
  }
  /********* End Utils *********/


  /********* Scripts *********/
  // Generate call graph for *specified function*
  // And write static analysis results of the function to JSON
  def methodCallGraph(methodName: String): Unit = {
    generateCallGraph(methodName)

    // --- Write functionInfoMap as JSON
    val jsonFile = s"./abstract/function_info_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      val fullPath = s"<your_hil_project_dir_prefix>/$file"
      val code = readSourceSnippet(fullPath, start, end)
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "is_atomic": ${name == methodName},
       |    "class": "${functionClsMap(name)}"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }

  // Generate call graph for *specified file*
  // Maintained static analysis info adds a Class field
  def fileCallGraph(fileName: String): Unit = {
    // Flow: File -> Classes -> Methods(in/not in classes)

    // All methods in the file
    val allMethods = cpg.file(fileName).method.name.filter(isReserved).toSet
    
    // Methods in classes
    var methodsInClass = Set[String]()
    val classes = cpg.file(fileName).typeDecl.filter{ typedecl =>
        typedecl.method.nonEmpty || typedecl.member.nonEmpty 
      }.filter(!_.name.contains("<")).filter(!_.name.contains("Exception")).toSet
    classesToAnalyze = classesToAnalyze ++ classes
    
    classes.foreach { cl => 
      val methods = cl.member.filter(_.dynamicTypeHintFullName.length!=0)
                                    .dynamicTypeHintFullName.flatMap(cpg.method.fullName(_))
                                    .name.filter(isReserved).toSet
      methodsInClass = methodsInClass ++ methods
      methods.foreach { method => 
        generateCallGraph(method)
        // Use minimal subclass to replace original class name
        val minimalClass = findMinimalSubclass(cl.name)
        functionClsMap(method) = minimalClass
      }
    }

    // Methods in file but not in classes
    val moduleLevelMethods = allMethods -- methodsInClass
    moduleLevelMethods.foreach { method => 
      generateCallGraph(method)
      functionClsMap(method) = ""
    }
  }

  // Batch generate call graph for *file collection*
  // And write static analysis results of the function to JSON
  def batchFileCallGraph(files: Set[String]): Unit = {
    files.foreach { file => 
      fileCallGraph(file)
    }

    // --- Write functionInfoMap as JSON
    val jsonFile = s"./abstract/function_info_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")
    
    val entries = functionInfoMap.map { case (name, (file, start, end)) => 
      val fullPath = s"<your_hil_project_dir_prefix>/$file"
      val code = readSourceSnippet(fullPath, start, end)
      val is_atomic = files.contains(file)
      val cls = functionClsMap.getOrElse(name, "")
      s"""  "$name": {
       |    "file": "$file",
       |    "lineNumber": $start,
       |    "lineNumberEnd": $end,
       |    "code": "$code",
       |    "is_atomic": $is_atomic,
       |    "class": "$cls"
       |  }""".stripMargin
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Function info map written to $jsonFile")
  }

  // Generate Attribute-Class mapping for *specified class*
  // And write to JSON
  def attrMapCls(targetClass: TypeDecl): Unit = {
    // Collect all classes in the inheritance chain
    val inheritanceChain = ArrayBuffer[TypeDecl]()
    var currentCls = targetClass
    inheritanceChain += currentCls
    
    // Bottom-up collection of all parent classes
    var continueLoop = true
    while (currentCls.inheritsFromTypeFullName.length != 0 && continueLoop) {
      val parentClsOpt = currentCls.inheritsFromTypeFullName.flatMap(cpg.typeDecl.fullName(_)).headOption
      parentClsOpt match {
        case Some(parentCls) =>
          inheritanceChain += parentCls
          currentCls = parentCls
        case None =>
          println(s"Warning: parent class not found for class ${currentCls.name}")
          continueLoop = false  // Break loop
      }
    }

    // If inheritance chain length > 1, add to global inheritance chain set
    if (inheritanceChain.length > 1) {
      allInheritanceChains += inheritanceChain.map(_.name)
    }

    // Inheritance handling: from root parent to subclass, accumulate attributes step-by-step - O(h x a)
    // Start processing from top-level parent (last element of reverse)
    for (i <- inheritanceChain.indices.reverse) {
      val currentClass = inheritanceChain(i)
      val currentClassName = currentClass.name
      
      // Get direct attributes of current class
      getAttrAssignments(currentClass)
      
      // If not root parent, inherit all attributes from direct parent
      if (i < inheritanceChain.length - 1) {
        val parentClass = inheritanceChain(i + 1)  // Fix: Parent index should be i+1
        val parentClassName = parentClass.name
        
        // Ensure current class has corresponding inner map in the map
        if (!attrClsMap.contains(currentClassName)) {
          attrClsMap(currentClassName) = MutableMap[String, String]()
        }
        
        // Inherit direct parent's attributes (parent attributes already include upper inheritance)
        if (attrClsMap.contains(parentClassName)) {
          // attrClsMap(parentClassName).foreach { case (creatingCls, attr) =>
          attrClsMap(parentClassName).foreach { case (attr, creatingCls) =>
            // Subclass attributes take precedence; inherit only if current class has no same-named attribute
            if (!attrClsMap(currentClassName).contains(attr)) {
              // Use minimal subclass to replace original creating class
              val minimalCreatingCls = findMinimalSubclass(creatingCls)
              attrClsMap(currentClassName)(attr) = minimalCreatingCls
            }
          }
        }
      }
    }
  }

    /******** Attribute Tree Builder *********/
  // Build complete attribute call tree from project_interface
  def buildAttributeCallTree(): MutableMap[String, String] = {
    val callPaths = MutableMap[String, String]()
    val visited = MutableSet[String]()  // Prevent circular references
    
    // Recursively build call paths
    def buildPaths(currentClass: String, currentPath: String, depth: Int): Unit = {
      // Prevent circular references and excessive recursion depth
      if (visited.contains(currentClass) || depth > 10) {
        return
      }
      
      visited += currentClass
      
      // Get attribute mapping of current class
      attrClsMap.get(currentClass) match {
        case Some(attributes) =>
          attributes.foreach { case (attributeName, targetClass) =>
            val newPath = if (currentPath.isEmpty) {
              attributeName
            } else {
              s"$currentPath.$attributeName"
            }
            
            // If targetClass is empty string, it indicates a direct attribute
            val resolvedTargetClass = if (targetClass.isEmpty) {
              s"${currentClass}_${attributeName}"  // Create virtual class name
            } else {
              targetClass
            }
            
            // Record call path (choose shorter path)
            if (!callPaths.contains(resolvedTargetClass) || callPaths(resolvedTargetClass).length > newPath.length) {
              callPaths(resolvedTargetClass) = newPath
            }
            
            // Continue recursive build (only for non-empty targetClass that is not self)
            if (targetClass.nonEmpty && targetClass != currentClass) {
              buildPaths(targetClass, newPath, depth + 1)
            }
          }
        case None =>
          // Current class has no defined attributes
      }
      
      visited -= currentClass
    }
    
    // Start building from project_interface
    buildPaths(project_interface, "", 0)
    
    callPaths
  }
  
  // Generate enhanced attribute mapping (includes complete call paths)
  def generateEnhancedAttrMapping(callPaths: MutableMap[String, String]): MutableMap[String, String] = {
    val enhancedMap = MutableMap[String, String]()
    
    // Add direct attributes
    attrClsMap.get(project_interface) match {
      case Some(directAttrs) =>
        // directAttrs.foreach { case (targetClass, attrName) =>
        directAttrs.foreach { case (attrName, targetClass) =>
          enhancedMap(targetClass) = attrName
        }
      case None =>
        println(s"Warning: No direct attributes found for $project_interface")
    }
    
    // Add all reachable classes and their call paths
    callPaths.foreach { case (targetClass, callPath) =>
      // Avoid duplicate addition of direct attributes (choose shorter path)
      if (!enhancedMap.contains(targetClass) || enhancedMap(targetClass).length > callPath.length) {
        enhancedMap(targetClass) = callPath
      }
    }
    
    enhancedMap
  }
  
  // Save enhanced attribute call tree
  def saveEnhancedAttrTree(enhancedMap: MutableMap[String, String], outputPath: String): Unit = {
    val jsonWriter = new BufferedWriter(new FileWriter(outputPath))
    jsonWriter.write("{\n")
    
    jsonWriter.write(s"""  "$project_interface": {\n""")
    
    val entries = enhancedMap.map { case (targetClass, callPath) =>
      s"""    "$targetClass": "$callPath""""
    }
    jsonWriter.write(entries.mkString(",\n"))
    
    jsonWriter.write("\n  }\n}")
    jsonWriter.close()
    
    println(s"Enhanced attribute call tree saved to: $outputPath")
  }
  /******** End Attribute Tree Builder *********/

  // Batch generate Attribute-Class mapping for *class collection*
  // And write to JSON, while building attribute call tree
  def batchAttrMapCls(files: Set[String]): Unit = {
    files.foreach { file => 
      val classes = cpg.file(file).typeDecl.filter{ typedecl =>
        typedecl.method.nonEmpty || typedecl.member.nonEmpty 
      }.filter(!_.name.contains("<")).filter(!_.name.contains("Exception")).toSet
      if (classes.length != 0) {
        classes.foreach { cl => attrMapCls(cl) }
      }
    }

    // --- Write attrClsMap as JSON
    val jsonFile = s"./abstract/attr_cls_map.json"
    val jsonWriter = new BufferedWriter(new FileWriter(jsonFile))
    jsonWriter.write("{\n")

    val entries = attrClsMap.map { case (clsName, attrMap) => 
      val innerEntries = attrMap.map { case (attr, creatingCls) =>
        s"""    "$attr": "$creatingCls""""
      }.mkString(",\n")
      
      s"""  "$clsName": {\n$innerEntries }"""
    }
    jsonWriter.write(entries.mkString(",\n"))
    jsonWriter.write("\n}\n")
    jsonWriter.close()

    println(s"Attribute-Class map written to $jsonFile")
    
    // --- Build attribute call tree
    println(s"\n  Building attribute call tree starting from $project_interface...")
    val callPaths = buildAttributeCallTree()
    val enhancedMap = generateEnhancedAttrMapping(callPaths)
    
    // Save enhanced attribute call tree
    val enhancedTreeFile = s"./abstract/enhanced_attr_tree.json"
    saveEnhancedAttrTree(enhancedMap, enhancedTreeFile)
    
    println(s"\n Attribute call tree construction completed!\n")
  }

  /********* End Scripts *********/


  /********* Main *********/
  def main(): Unit = {
    // Project Closure generated by python utils
    val files = readFilesFromJson("./tarFiles.json")
    // Generate attribute-class map
    // Output: attr_cls_map.json
    batchAttrMapCls(files)

    // Generate call graph for all files
    // Output: function_info_map.json
    batchFileCallGraph(files)
  }

}

