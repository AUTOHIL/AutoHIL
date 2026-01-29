# AutoHIL-2026

We propose AUTOHIL, an LLM-based framework designed for the end-to-end automation of ECU system functional testing on HIL benches through domain knowledge augmentation. 

To achieve general adaption across different HIL benches, AUTOHIL automatically analyzes THPs of HIL bench to build a THP knowledge base, allowing the LLM to retrieve and utilize specific bench primitives via RAG. To complement the tacit domain knowledge involved in the requirements, we extract domain concepts and functional logic from AUTOSAR ECU source code and configuration files to augment system functional requirements. Finally, to realize end-to-end test automation, we use the augmented requirements as input and the THP Knowledge Base as the retrieval source, and utilize LLM to decompose requirements into test steps aligned with THPs, format steps into behavior trees as an intermediate representation, and ultimately converts them into executable HIL test scripts.

AUTOHIL targets in-house testing. We assume that testers, as first-party developers, can access HIL bench interfaces and AUTOSAR C source code of the ECU under test. This assumption aligns with the industry trend of tighter integration between testing and development roles. In practice, test engineers need HIL interfaces to write scripts, and also inspect source code to clarify implementations and localize defects. AUTOHIL leverages this reality by turning scattered project knowledge into structured information consumable by LLMs.

<img src=".\picture\overview.png"  />

### Environment Setup

##### Hardware environment

<img src=".\picture\hardware_setup.png" alt="hardware_setup" style="zoom: 33%;" />

Figure above illustrates the hardware configuration in our evaluation. It includes a PC workstation running our framework, an ECU under test, a HIL bench for HIL testing, a PCAN interface for CAN communication and a Lauterbach debugger for monitoring internal key variable trace.

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

#### Functional Correctness: Implementation Detail of *Variable Trace Equivalence* Analysis

<img src=".\picture\functional_correctness.png" alt="functional_correctness"  />

To assess whether the automatically generated test scripts achieve the same testing efficacy as expert-written manual scripts, we introduce a novel metric, **Variable Trace Equivalence Rate**. We first need to determine the key variables for each requirement. A manual approach would require carefully reading and understanding the source code to locate key variables, which is time-consuming and labor-intensive. Therefore, we use an automated method to locate key variables from requirements. Next, we use a debugger to connect to the ECU debug interface and trace the value of variables. We export the time-series data log to a PC through a USB connection. Finally, we compare the similarity of signal sequences on the PC to obtain a Variable Trace Equivalence rate for each requirement. When the rate exceeds a threshold, we consider the impacts of the two scripts to be equivalent. The procedure is as follows.

##### a. Automatic Identification of Key Variables.

For requirements in different function modules, we let the LLM parse requirement semantics based on the module call graph and the graph node function summaries, locate which functions implement the requirement, and then extract variables from these relevant functions. We denote this set as `var_source`.

Next, we parse the `DWARF` debugging information and the Symbol Table from the `ELF` file. DWARF is a standard format for storing program debugging information (includes variable names, types, storage locations, function names, parameter lists, and line numbers in the source code.) It is mainly used by debuggers to build the mapping between binary instructions and the original source code, enabling source-level variable inspection. The Symbol Table stores the mapping between all global and static variables and their corresponding memory addresses (or offsets). It is used by the linker to resolve symbol references and determine the final storage locations of symbols. Since DWARF also records local variables, we select the global and static variables that have definitive memory addresses in both DWARF and the Symbol Table, and denote this set as `var_debug`.
Finally, we take the intersection of `var_source` and `var_debug`. The variables in the intersection are the key variables to be monitored for the requirement.

##### b. Monitoring Internal ECU Variables via a Debugger.

We use a Lauterbach debugger to connect to the ECU's debug interface. The Lauterbach debugger provides Trace functionality in three modes. The first mode is Onchip Trace, which stores trace data in on-chip memory, but most chips only have 1K--2MB internal space. The second mode is Offchip Trace, which stores trace data in external Lauterbach trace hardware (e.g., PowerTrace). Both modes suffer from limited storage, and require the chip to support ETM and ETB/ETF/ETR functionalities. Therefore, we choose the third mode, Snooper Sampling. It relies on the runtime memory access path provided by the chip to sample memory and obtain variable values. In this mode, the debugger periodically samples the values of multiple monitored variables. After stopping the chip, it aggregates the time-series data log for analysis. The sampling duration and capacity are not limited. Most chips can support it directly, and it can be implemented with standard debugger models, making it highly suitable for recording variable traces. We set the expected sampling frequency to 0.5 ms, since ECU tasks update data with periods ranging from 0.5 ms to 25 ms. This sampling frequency can be configured in Lauterbach PowerView software. We then use Python to control the Lauterbach debugger and configure trace commands to automate monitoring and tracing.

Since the time-series data log is analyzed offline, it is difficult to identify which data segment corresponds to the execution of a given test script. To solve this problem, we insert *Test Marker Frames* at the start and end of script execution. We select a high-priority CAN ID (0x002) that does not involve normal ECU functions as the marker frame ID. At script start, the marker frame payload is set to the requirement ID, while at script end, the payload is set to all 0xFF. We then use PCAN to capture the timestamps of these two CAN messages as the start and end timestamps. With these timestamps, we slice the corresponding time-series segment from the log. This effectively maps each test script to its associated time-series data segment.

##### c. Variable Trace Equivalence Comparison Algorithm.

For a specific requirement $R$, suppose there is a set of $m$ key variables $V=\{V_1,V_2,\ldots,V_m\}$. For each key variable $V_i$, we compare its time-series data segments collected during execution of the two scripts (manual script vs generated script). We use our `TraceCompare` algorithm to compute a variable trace similarity score $S_i$ for each variable, and then aggregate the scores of multiple variables to obtain the final Variable Trace Equivalence Rate ($ER$) for the specific requirement.

For a single variable, the manual script and the generated script produce two time-series traces $(t_1,x_1)$ and $(t_2,x_2)$. The similarity score $S_i$ is computed by `TraceCompare` as follows.

We first perform time alignment. Since timestamps from two sampling sessions often differ, we use linear interpolation to resample both curves onto a unified relative time axis $T$, yielding vectors: $\mathbf{X}_1 = [x_{1,1}, ..., x_{1,N}]$ and $\mathbf{X}_2 = [x_{2,1}, ..., x_{2,N}]$.
We apply a moving average filter to reduce high-frequency noise that can affect shape computation: $\mathbf{X}_{smooth} = \text{MovingAverage}(\mathbf{X}, k)$, where $k$ is the window size. We then apply z-normalization to both the raw sequences $\mathbf{X}_1, \mathbf{X}_2$ and the smoothed sequences $\mathbf{X}^{smooth}_1, \mathbf{X}^{smooth}_2$. For any sequence $\mathbf{x}$, the $i$-th point of its normalized sequence $\mathbf{z}$ is calculated as: $z_i = \frac{x_i - \mu}{\sigma + \epsilon}$, where $\mu$ is the mean, $\sigma$ is the standard deviation, and $\epsilon$ is a minimum value to prevent division by zero.

The similarity score of a single variable is a robust hybrid metric composed of three weighted sub-scores.

First is the **rigid score**. It evaluates how well the two normalized traces $\mathbf{Z}_1, \mathbf{Z}_2$ overlap on the original time axis. We compute the *Pearson correlation coefficient* $r = \text{Pearson}(\mathbf{Z}_1, \mathbf{Z}_2)$ and map it linearly to $[0, 1]$ as the correlation score: $S_{corr} = \frac{r+1}{2}$, assessing the correlation of the overall trend. Simultaneously, we calculate the Root Mean Square Error: $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (z_{1,i}-z_{2,i})^2}$, representing the Euclidean distance between points, and convert it to a 0-1 score via an exponential decay function: $S_{rmse} = e^{-RMSE/\gamma}$, where $\gamma$ is a sensitivity parameter (default 0.5). The final Rigid Score is the average: 
$$
S_{rigid} = 0.5 \times S_{corr} + 0.5 \times S_{rmse}.
$$
**Shape Score** is the second sub-score to evaluates the shape similarity of the smoothed, normalized sequences  $\mathbf{Z}^{smooth}_1, \mathbf{Z}^{smooth}_2$ under allowed time stretching. Even if amplitudes and phases differ, similar trends yield high scores. The core algorithm employs DTW (Dynamic Time Warping) to find aligned points by stretching or compressing the time series. The normalized DTW distance is converted to a score: 
$$
S_{shape} = \frac{1}{1 + \text{DTW}(\mathbf{Z}^{smooth}_1, \mathbf{Z}^{smooth}_2)/\sqrt{N}}.
$$
The third sub-score is **EndState Score**, which evaluates whether the final states of the two smoothed traces $\mathbf{X}^{smooth}_1$ and $\mathbf{X}^{smooth}_2$ are equivalent. The end state may be a steady value $v$ or the average of the final trend window $W$: $v = \frac{1}{W}\sum_{j=N-W+1}^n x_j^{smooth}$. We compute the difference between the two end-state means and normalize it by the dynamic range $R$: $S_{end}=1-\frac{|v_1-v_2|}{R}$, where $R=\max(\mathbf{X}^{smooth}_1\cup\mathbf{X}^{smooth}_2)-\min(\mathbf{X}^{smooth}_1\cup\mathbf{X}^{smooth}_2).$

**The variable similarity score is the weighted sum of the three sub-rates**, with a range of $[0,1]$:
$$
S_i=\text{TraceCompare}(V_i)=(W_{rigid}\times S_{rigid}^{(i)})+(W_{shape}\times S_{shape}^{(i)})+(W_{end}\times S_{end}^{(i)}).
$$

To evaluate execution consistency at the requirement level, we must **aggregate** the scores of all key variables in the requirement. Assuming all key variables are equally important, we use the arithmetic mean. The resulting mean is the final requirement-level Equivalence Rate (ER), converted to a percentage:
$$
ER=\frac{1}{m}\sum_{i=1}^{m}S_i\times 100.
$$
When $ER$ exceeds a threshold (set to 75 empirically in our experiments), we consider that the generated script and the manual script are equivalent in terms of functional impact, and the requirement passes functional correctness evaluation.

##### d. Result Analysis.

Ultimately, we statistically analyzed the mean Variable Trace Equivalence Rates for requirements across different modules, as well as the proportion of requirements in each module that passed the functional correctness evaluation (i.e.,$ER \ge 75$).

<img src=".\picture\acu_var_trace_equiv.png" alt="acu_var_trace_equiv" style="zoom: 25%;" />

<img src=".\picture\asdm_var_trace_equiv.png" alt="asdm_var_trace_equiv" style="zoom: 25%;" />


We report the **A**verage Variable Trace **E**quivalence **R**ate (AER) and the functional correctness pass rate defined by the criterion $ER\ge75$ for each module. On ACU, the overall AER is 84.1 and the pass rate is 85.25% (526 out of 617 test scripts). On ASDM, the overall AER is 80.5 and the pass rate is 87.33% (889 out of 1018 test scripts). Modules differ in how well their traces align. These differences are mainly driven by logical determinism, disturbances from physical/signal simulation, and timing sensitivity in closed-loop behaviors.

Modules dominated by standardized protocols or discrete state machines tend to exhibit higher and more concentrated equivalence scores. For example, ACU GPO module (AER=95.5, 100% pass rate) is pure I/O control with limited external disturbances, so generated scripts can more easily reproduce the same variable evolution as the manual scripts. Similarly, the Diagnostic Event Manager (DEM) module is stable on both ECUs (ACU-DEM AER=91.5, ASDM-DEM AER=87.9), reflecting the high predictability of internal variable trajectories when DTC state machine boundaries are well-defined. The diagnostic module also shows high pass rates (ASDM-CanD 91.75%, ACU-CanD 86.89%). Its outliers mainly arise at timing boundaries such as pending handling and timeout windows, rather than in the main protocol flow.

Modules that involve analog signal injection or physical-environment simulation typically exhibit moderate AER and more pronounced dispersion with long-tail cases. For instance, power and diagnostic control modules fall into a lower range on both ECUs (ACU-PWR AER=75.8, ASDM-PWR AER=76.2; ACU-DCS AER=76.2, ASDM-DCS AER=77.6). These gaps are often caused by waveform edges, ripple, filtering effects, and noise near thresholds. Even with the same test objective, small simulation errors can shift threshold-crossing instants or alter branch decisions.

The lowest AER are typically observed in highly timing-sensitive modules or high-dynamics modules that including closed-loop perception-control chains, which are more sensitive to timing and initial conditions. Small delays or state deviations can be amplified downstream, leading to larger trace divergence. For example, the ACU EDR module (AER=71.5) is highly sensitive to millisecond-level alignment of trigger windows. On ASDM, AEB (AER=70.7) is highly sensitive to timestamps and delay budgets near TTC (Time to Collision) critical triggering. ObjectFusion (AER=72.9) and LKA/LCC (AER=75.1) are also prone to trace jumps due to association branching, ID maintenance, or phase misalignment in closed-loop control. These long-tail and low-score cases indicate that, in continuous closed-loop systems, timing and state management remain a key bottleneck for high-fidelity reproduction in automatically generated tests.

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

#### Failure Detection Capability

The generated scripts exhibit clear failure detection capability on both ECUs. These newly revealed failures have been explicitly confirmed as valid defects by our partner supplier and are currently undergoing remediation. We analyzed these new defects and found that they generally align closely with the mechanism of source-driven requirement augmentation. They can be further grouped into three categories. We analyze each category and provide representative defect examples.

##### a. Implementation Detail Defects.

AUTOHIL reads the concrete implementation logic during source analysis and uncovers defects and bugs at the implementation level. 

For example, on the ACU, when a Left-Crash (Non-fire) event is followed 2 seconds later by a Front-Crash (Fire/Level 4) event, the ACU incorrectly broadcasts the Data Value (DV) of the initial non-fire event. Source analysis reveals that the EDR utilizes a double-buffer mechanism (BUF0, BUF1). However, the logic lacks a mechanism to toggle the reading index to the currently active buffer containing the fire event data. While manual scripts typically test single events, the framework, aware of the buffer switching logic, generates this specific interleaved event sequence.

On the ASDM, when the Region2 blockage timer reaches 179s and the obstruction is removed, Core0 attempts to reset `timer=0` while Core1 simultaneously executes an increment `timer++`. The reset operation is overwritten, causing the timer to reach the threshold in the next cycle. AUTOHIL identifies this vulnerability by detecting unprotected access to the global array `blockageRegionTimer` across multiple cores, generating a test case that manipulated event timing to trigger this atomicity violation.

##### b. Corner Case and Conflict Handling.

AUTOHIL analyzes the if-else and switch-default branches in the source code to generates conflict tests and boundary combinations where multiple conditions hold simultaneously. 

For example, on the ACU, when the PAB (Passenger Airbag) is configured via UDS service 0x2E as a Hard Switch (suppressing fault reporting), receiving a conflicting Soft Switch CAN signal from the ICM should trigger a `PADIMGR_PAB_SOFT_CFG` fault. The ECU fails to report this. AUTOHIL identifies the conflicting logic branches in the source code and generates a test case that simultaneously satisfied both mutually exclusive conditions.

On the ASDM, during high-speed driving, the lane marking is occluded for 0.9 seconds and then recovers. The output lateral distance jumps back to the old position from 0.9 seconds earlier, and the LKA produces incorrect torque due to state residue in the circular buffer. Analysis reveals a mismatch: the buffer capacity is 30 frames, but the reset threshold is 22 frames. Losing 20 frames freezes the frameCounter without triggering a reset. Upon restoration, 1 frame of new data is averaged with 15 frames of stale data. Manual scripts only test "Continuous Recognition" or "Loss > 1s," whereas AUTOHIL, detecting the threshold-capacity mismatch, automatically targets the 0.5s--0.95s boundary window.

##### c. State Transition and Lifecycle Errors.

By parsing internal state machines, AUTOHIL identifies unreachable states or invalid transition paths caused by variable lifecycle mismanagement. 

For example, on the ACU, a GPO `STB` fault fails to recover when transitioning to the `STG` state. Fault clearance requires the channel to be ON with specific bits cleared. However, when the script rapidly switches from STB to STG, the debounce counter is not cleared due to the fast transition, preventing the clearance condition from being met. While manual scripts focus on "Golden Paths," AUTOHIL analyzes the`GpoDrvr` state machine and generates a rapid transition test that exposes this debounce handling error. 

ASDM fails to retry image transmission after a timeout. If the VP response times out (2s) during image reading, a retry request returns "Read Failed" instead of restarting transmission. Source analysis shows that the timeout handler resets the global state `imgCapCtx.state` but fails to reset `isStart`, which is only cleared upon a successful pipeline. Consequently, the retry logic incorrectly perceives `isStart=True` and enters the error branch.

#### Ablation Study

```bash
> nohup python ./eval/ablation/wo_api_knowledge_base.py > log/process.log 2>&1 &
> nohup python ./eval/ablation/wo_req_augment.py > log/process.log 2>&1 &
> nohup python ./eval/ablation/wo_req_decompose.py > log/process.log 2>&1 &
```

