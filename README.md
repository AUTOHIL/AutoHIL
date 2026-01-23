# AutoHIL-2026

We propose AUTOHIL, an LLM-based framework designed for the end-to-end automation of ECU system functional testing on HIL benches through domain knowledge augmentation. 

To achieve general adaption across different HIL benches, AUTOHIL automatically analyzes THPs of HIL bench to build a THP knowledge base, allowing the LLM to retrieve and utilize specific bench primitives via RAG. To complement the tacit domain knowledge involved in the requirements, we extract domain concepts and functional logic from AUTOSAR ECU source code and configuration files to augment system functional requirements. Finally, to realize end-to-end test automation, we use the augmented requirements as input and the THP Knowledge Base as the retrieval source, and utilize LLM to decompose requirements into test steps aligned with THPs, format steps into behavior trees as an intermediate representation, and ultimately converts them into executable HIL test scripts.

AUTOHIL targets in-house testing. We assume that testers, as first-party developers, can access HIL bench interfaces and AUTOSAR C source code of the ECU under test. This assumption aligns with the industry trend of tighter integration between testing and development roles. In practice, test engineers need HIL interfaces to write scripts, and also inspect source code to clarify implementations and localize defects. AUTOHIL leverages this reality by turning scattered project knowledge into structured information consumable by LLMs.

<img src=".\picture\overview.png"  />

### Environment Setup

##### Hardware environment

<img src=".\picture\hardware_setup.png" alt="hardware_setup" style="zoom: 33%;" />

##### Software environment

1. Setup Python environment

    ```bash
    > conda create -n autohil python=3.12
    > conda activate autohil
    > pip install -r requirements.txt
    ```

2. Install [Joern](https://docs.joern.io/installation/) for static analysis.

##### Folder Structure

```
│  config.json				# user Configuration
│  init_db.py				# initialize rag database
│  main.py					# main end-to-end test generation pipeline
│  
├─apiPool
│      getTars.py			# get sub-package for HIL project
│      CGGenerator.scala	# THP Analyzer based on Joern
│      genAbstract.py		# helper functions for mergeAndTopo.py			
│      mergeAndTopo.py		# THP summarizer
│      
├─reqAugmentation
│  │  reqExtractor.scala	# AUTOSAR Functional Modular Anaylzer
│  │  mergeAndTopo.py		# generate modular summaries
│  │  reqExtractor.py		# construct requirement-point corpus
│  │  reqAugment.py			# requirement augmenter
│  │  
│  └─depClosureUtils
│          genAbstract.py	# helper functions for mergeAndTopo.py
│          getCFileDeps.py	# get modular sub-package for ECU source code
│          merge_dot_advanced.py	# for divide-and-conquer
│          merge_function_maps.py
|
├─reqDecomposer
│      btEncoder.py			# behavior tree generator
│      reqDec_local_rag.py	# requirement decomposer
|
├─IR2Script
│      btDecoder.py			# rule-based transformation
│      post_process_rag.py 	# LLM-based post-process
|
├─eval						# Evaluation
│  ├─ablation				# ablation study
│  │      reqDec_wo_rag.py
│  │      wo_api_knowledge_base.py
│  │      wo_req_augment.py
│  │      wo_req_decompose.py
│  │      
│  ├─argsFinding			# for functional correctness
│  │  │  dwarfParser.py
│  │  │  getRelatedSrc.py
│  │  │  matchRelArgs.py
│  │  │  
│  │  └─abstract
│  │          genAbstract.py
│  │          srcCode.scala
│  │          
│  └─debugger
│       trace_fusion.py
│              
└─utils
        api_loader.py		# utils for btDecoder.py
        behaviorTree.py		# definition of behavior tree IR
        excel_parser.py		# utils for parsing excel
```

### Prerequisite Preparation

This phase includes **THP Knowledge Base construction** and **Requirement-point Corpus construction**.

Input:

- The set of THPs (HIL-bench API)
- AUTOSAR ECU source code


##### THP Knowledge Base
```bash
> cd apiPool
```

1. Run `python getTars.py` to get sub-package files listed in  `tarFiles.json` .

2. Run **Joern**, and execute commands below to generate:

   - CPG bin file (`/cpg/cpg.bin`)
   - sub-call-graphs in `callgraph/`
   - `functionInfoMap.json` and `enhanced_attr_tree.json`, which store all Static Metadata and Attribute Access Path

   ```shell
   # In bash
   > joern-parse <your_hil_project_dir> --output ./cpg/cpg.bin
   # In the Joern shell
   Joern> :load ./CGGenerator.scala
   Joern> CallGraphGenerater.main()
   ```

3. Then run commands below to generate summaries for THPs (`/abstract/api_abstract.json`)

   ```bash
   > nohup python mergeAndTopo.py > log/summarizer.log 2>&1 &
   ```

##### Requirement-point Corpus

```bash
> cd reqAugmentation
```

1. Run `getCFileDeps.py` to get sub-package files listed in `cTarFiles.json`

2. Run **Joern**, and execute commands below:

   * In cases where the ECU codebase is too large for Joern to process effectively, we employ a divide-and-conquer strategy. (Execute below twice with different directories)

   ```bash
   # Divide
   # In bash
   > c2cpg.sh -J-Xmx202080m <your_ecu_source_dir1> --output --include <autosar_header_filepaths> ./cpg/cpg1.bin
   # In the Joern shell
   ### a. add overlay
   Joern> ImportCpg("./cpg/cpg1.bin")
   Joern> save
   ### b. get static metadata
   Joern> :load reqExtract.scala
   Joern> CallGraphGenerator.batchCallGraph()
   
   # Conquer
   # In bash
   > python merge_dot_advanced.py
   > python merge_function_maps.py
   ```

3. Then run commands below to generate summaries for ECU FUNCTION modules

   ```bash
   > nohup python mergeAndTopo.py > log/reqExtract.log 2>&1 &
   ```

4. Run `python reqExtractor.py` to extract requirement-points (`./req_corpus/FUNCTION_corpus.json`)

##### Final construction step

```bash
> cd ../autohil
> python init_db.py
```

### End-to-End Automation Pipeline

Including **Requirement Augmentation**, **Requirement Decomposition** and **IR-to-Script Generation**.

```bash
> nohup python main.py > log/process.log 2>&1 &
```

The generated test scripts are stored in the corresponding FUNCTION module folders under the `IR2Script/` directory.

### Evaluation

##### Functional Correctness

<img src=".\picture\functional_correctness.png" alt="functional_correctness"  />

1. Identify key variables: 

   Input:  your `.elf` file

   ```shell
   # a. Find out key variables of each requirements
   > cd eval/argsFinding
   Joern> :load ./abstract/srcCode.scala
   Joern> scrCodeAnalyzer.main()
   > python genAbstract.py
   > python getRelatedSrc.py
   # b. Parse dwarf
   > python dwarfParser.py
   # c. Intersect
   > python matchRelArgs.py
   ```

2. Monitor using debugger

3. Analyze: 

   ```bash
   > cd eval/debugger
   > python trace_fusion.py \
       --csv1 <var_csv_of_generated_script> \
       --csv2 <var_csv_of_manual_script> \
       --symbol "g_Engine_RPM" \
       --plot \
       --outdir "./figs"
   ```

##### Ablation Study

```bash
> nohup python ./eval/ablation/wo_api_knowledge_base.py > log/process.log 2>&1 &
> nohup python ./eval/ablation/wo_req_augment.py > log/process.log 2>&1 &
> nohup python ./eval/ablation/wo_req_decompose.py > log/process.log 2>&1 &
```

