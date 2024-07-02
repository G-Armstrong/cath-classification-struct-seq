import pandas as pd
import ast
import requests
import Bio

from Bio import SeqIO
from io import StringIO
from Bio.PDB.Polypeptide import protein_letters_3to1

def extract_resid_ranges(pdb_filename, threshold=3):
    residues = []
    
    # Read the PDB file and extract residue IDs
    with open(pdb_filename, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                resid = int(line[22:26].strip())
                residues.append(resid)
    
    # Find unique and sorted residue IDs
    unique_residues = sorted(set(residues))
    
    # Find contiguous ranges with a gap threshold provided by the user (defaults to 3)
    ranges = []
    start = unique_residues[0]
    end = unique_residues[0]

    for i in range(1, len(unique_residues)):
        if unique_residues[i] - end <= threshold:
            end = unique_residues[i]
        else:
            ranges.append((start, end))
            start = unique_residues[i]
            end = unique_residues[i]

    ranges.append((start, end))

    return ranges


def get_sequence_from_pdb(pdb_filename):
    """
    The sequences come from the PDB files
    """
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
    assert len(structure) == 1

    seq = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # This checks if it's a standard residue
                    seq.append(protein_letters_3to1[residue.get_resname()])
                else:
                    print('nonstandard', residue.get_id())

    return ''.join(seq)

def get_uniprot_accession_from_pdb(pdb_code):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code.lower()}"
    response = requests.get(url)
    response.raise_for_status()
    pdb_data = response.json()

    try:
        uniprot_id = list(pdb_data[pdb_code.lower()]['UniProt'].keys())[0]
        return uniprot_id
    except KeyError:
        print(f"UniProt ID not found for PDB code {pdb_code}")
        return None

def get_fasta_sequence_from_uniprot(uniprot_accession):
    """
    Fetches the amino acid FASTA sequence from UniProt given a UniProt accession.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_accession}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        fasta_content = response.text
        fasta_io = StringIO(fasta_content)
        record = SeqIO.read(fasta_io, "fasta")
        return str(record.seq)
    else:
        raise Exception(f"Failed to retrieve FASTA sequence for UniProt accession {uniprot_accession}. Status code: {response.status_code}")
        
def safe_eval(val):
    """
    Function to safely convert string representation of list of tuples to actual list of tuples
    """
    if pd.isna(val):
        return val
    else:
        return ast.literal_eval(val)