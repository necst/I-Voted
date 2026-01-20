# I-Voted
Neural Processing Unit Automation

## How To Clone
To clone this repository: 

```bash
git clone --recursive https://github.com/tuo-username/I-Voted.git
```
Alternatively, if the reposititory is already cloned

```bash
git submodule update --init --recursive
```

Instead, to update the submodule only: 

```bash
git submodule update --remote mlir-aie
```

## How To Use

The repository is still under development and now intended for internal usage only. Several bugs or possible optimizations may exist. Please, forward any improvement suggestiond :D 

The main idea is to use mlir-aie from a private independent repository. 
To do so, the repository uses a modified version of mlir-aie Makefile that allows to build your design even if you are outside MLIR-AIE

### Step 1: source XRT
```bash
source /opt/xilinx/xrt/setup.sh
```

### Step 2: Prepare the environmnet
```bash
cd mlir-aie
source ./utils/quick_setup.sh
```

This command will give you a terminal named (ironenv). Now you can move in the application folder

### Step 3: Export MLIR_AIE_PATH
```bash
export MLIR_AIE_PATH=<your path to> mlir-aie
```

notice that, if you do not define a new position for MLIR_AIE_PATH, this repository assumes the submodule, thus having mlir-aie as I-Voted subfolder

### Step 4: Build Application
```bash
cd ivoted
make
```

### Step 5: Run the Application
C++ Version: 

```bash
make run
```

Python Version

```bash
make run_py
```



