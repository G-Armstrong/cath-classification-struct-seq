# medbio-challenge
protein structure classification using the CATH database.

**Important**:
There are many structure and sequence files in /data. When pulling this repository, make sure you increase Git's default buffer size with the following command and then try cloning:
```bash
git config --global http.postBuffer 524288000
```

**General Workflow**:
 `DataProcessing.ipynb` -> (Remodeling to correct sequence gaps) -> `GraphDataset.ipynb` -> `python models/train.py`

**Disclaimer**:
Some files contained within data/*/pdb were remodeled using SCWRL4. While, in many cases, these files are not a complete representation of the starting structure and, therefore, were not used (SCWRL4 is good for modeling sequence substituions, not insertions), it is important to read and understand the [usage guidelines](http://dunbrack.fccc.edu/lab/scwrl) for SCWRL4 outputs. The current implementation DOES NOT utilize SCWRL4 remodeled structures. These file outputs are included for experimentation purposes only. Instead, the starting structures for all cath_ids were utilized when creating graphical representaitons.  

## Setup Instructions

### 1. Create and Activate Conda Environment (on MacOS/Linux)

Create a conda environment called `medbio-challenge` with Python 3.11.9 and install dependencies:

```bash
conda create -n medbio-challenge python=3.11.9 -y
conda activate medbio-challenge
pip install -r src/requirements.txt
```


### 2. Install `ncbi-blast+` for Sequence Alignment
 
`ncbi-blast+` is not available in Conda channels. It needs to be installed manually.
If you are utilizing modified sequences in `data/*/seqs/modified.fasta`, continue to Step 3. This requirement is necessary for a full rerun of `notebooks/DataProcessing.ipynb`

a. **Download `ncbi-blast+`**:

The compressed archive for the x86_64 (64-bit Intel/AMD) linux distribution is provided at the following link:

Visit the [NCBI website](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) to download alternate versions of `ncbi-blast+` as a compressed archive (`tar.gz`).

b. **Extract the Archive**:

`ncbi-blast+` does not require a traditional configure and make process for installation from the extracted archive. Instead, you can directly use the binaries (blastp, blastn, etc.) provided in the bin/ directory after extraction.

Extract the downloaded archive to a location within your Conda environment.
Replace /path/to/your/conda/env with the actual path to your Conda environment's directory (e.g., envs/medbio-challenge)
For example:
```bash
tar -xzvf ncbi-blast-2.15.0+-x64-linux.tar.gz -C /path/to/your/conda/env
```

### 3. **Set Up Environment Variable**
```bash
vi ~/.bashrc
export PATH="/path/to/your/conda/env/ncbi-blast-2.15.0+/bin:$PATH"
:wq
source ~/.bashrc
```
### 4. Acticate env and confirm installation
```bash
conda activate medbio-challenge 
blastp -h
```
### 5. Install PyRosetta (Optional)

PyRosetta can be used for sequence remodeling. Make sure your env is active.

```bash
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
pip install pyrosetta-distributed
conda config --add channels https://conda.graylab.jhu.edu
conda install pyrosetta	
```

## Notebooks
### After activating the conda env, notebooks can be accessed by typing `jupyter notebook` into the console and hitting ENTER - Ctrl + C / Ctrl + V one of the provided links into your web browser
The `/notebooks` dir contains all code for data preprocessing

