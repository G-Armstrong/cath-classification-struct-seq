# medbio-challenge
protein structure classification using the CATH database.


## How to set up conda environment (on MacOS/Linux)

1. Create a conda environment called `medbio-challenge` with Python 3.11.9 install dependencies specified by src/requirements.txt.
```bash
conda create -n medbio-challenge python=3.11.9 -y && \
conda run -n medbio-challenge pip install -r src/requirements.txt
```

2. Installing `ncbi-blast+` for Sequence Alignment
 
`ncbi-blast+` is not available in Conda channels. It needs to be installed manually.

a. **Download `ncbi-blast+`**:

The compressed archive for the x86_64 (64-bit Intel/AMD) linux distribution is provided in this repo.

Visit the [NCBI website](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) to download alternate versions of `ncbi-blast+` as a compressed archive (`tar.gz`).

b. **Extract the Archive**:
`ncbi-blast+` does not require a traditional configure and make process for installation from the extracted archive. Instead, you can directly use the binaries (blastp, blastn, etc.) provided in the bin/ directory after extraction.

Extract the downloaded archive to a location within your Conda environment.
Replace /path/to/your/conda/env with the actual path to your Conda environment's directory (e.g., envs/medbio-challenge)
For example:
```bash
tar -xzvf ncbi-blast-2.15.0+-x64-linux.tar.gz -C /path/to/your/conda/env
```

3. **Set Up Environment Variable**
```bash
vi ~/.bashrc
export PATH="/path/to/your/conda/env/ncbi-blast-2.15.0+/bin:$PATH"
:wq
source ~/.bashrc
```
4. Acticate env and confirm installation
```bash
conda activate medbio-challenge 
blastp -h
```

