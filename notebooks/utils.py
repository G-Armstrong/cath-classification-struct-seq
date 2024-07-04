import pandas as pd
import ast
import requests
import Bio
import subprocess
import xml.etree.ElementTree as ET

from Bio import SeqIO
from io import StringIO
from Bio.PDB.Polypeptide import protein_letters_3to1

from Bio.Align import PairwiseAligner

def extract_aligned_segment(given_sequence, ground_truth_sequence):
    """
    Extracts the segment from the ground truth sequence that most aligns with the given sequence.
    
    Args:
    given_sequence (str): The sequence to be aligned.
    ground_truth_sequence (str): The reference sequence to align against.
    
    Returns:
    str: The aligned segment from the ground truth sequence.
    """
    
    # Initialize the aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    
    # Perform the alignment
    alignments = aligner.align(given_sequence, ground_truth_sequence)
    
    # Get the best alignment
    best_alignment = alignments[0]
    
    # Extract the aligned sequences
    aligned_given = best_alignment.aligned[0]
    aligned_truth = best_alignment.aligned[1]
    
    # Identify the start and end positions in the ground_truth_sequence
    start_pos = aligned_truth[0][0]
    end_pos = aligned_truth[-1][1]
    
    # Extract the aligned segment from the ground_truth_sequence
    aligned_segment = ground_truth_sequence[start_pos:end_pos]
    
    # Print the aligned segment
    # print(f"Aligned segment from the ground truth sequence: {aligned_segment}")
    
    # Print the alignment for reference
    # print(best_alignment)
    
    return aligned_segment

def manual_alignment(query_seq, aligned_segment, subject_seq):
    query_length = len(query_seq)
    aligned_length = len(aligned_segment)
    subject_length = len(subject_seq)
    
    # Initialize a variable to store the maximum match found
    max_match_length = 0
    best_shifted_segment = aligned_segment
    
    # Try shifting aligned_segment to the right
    print('Trying a right shift of the aligned segment.')
    for shift in range(aligned_length):
        shifted_segment = '+' * shift + aligned_segment
        if len(shifted_segment) == query_length:  # We have added enough '+' characters
            # print('`aligned_segment` length now matches `query_seq` length\n')
            if find_consecutive_match(query_seq, shifted_segment, direction='right'):
                print('Right shift solve was successful.\n')
                break
    else:
        print('Trying a left shift of the aligned segment.')
        # If no match found by shifting right, try shifting left
        for shift in range(aligned_length):
            shifted_segment = aligned_segment + '+' * shift
            if len(shifted_segment) == query_length: # We have added enough '+' characters
                # print('`aligned_segment` length now matches `query_seq` length\n')
                if find_consecutive_match(query_seq, shifted_segment, direction='left'):
                    print('Left shift solve was successful.\n')
                    break

    print('A match between the shifted `aligned_segment` and the `query_seq` was found:')
    print(query_seq)
    print(shifted_segment)
    
    # Replace '+' with amino acids from subject_seq
    print("\nReplacing '+' values in the shifted `aligned_segment` with amino acids codes from the `subject_seq`.\n")

    # Find the alignment between the shifted `aligned_segment` and the `subject_seq`
    for shift in range(subject_length):
        final_alignment = ' ' * shift + shifted_segment
        # Confirm the alignment
        if find_consecutive_match(subject_seq, final_alignment):
            break

    # Replace '+' with corresponding elements from subject_amino_acids
    subject_amino_acids = list(subject_seq)
    final_alignment_amino_acids = list(final_alignment)
    modified = [subject_amino_acids[i] if char == '+' else char for i, char in enumerate(final_alignment_amino_acids)]
    
    # Remove ' ' entries
    filtered_final_alignment = [char for char in modified if char != ' ']
    
    # Combine into a single string
    repaired_segment = ''.join(filtered_final_alignment)

    print('Successfully repaired the aligned segment with the manual solve.')
    print(f"Repaired aligned segment length: {len(repaired_segment)}")

    return repaired_segment

def find_consecutive_match(macro_seq, micro_seq, direction=''):
    '''
    We always align the micro_seq to the macro_seq. The macro_seq contains more information than the micro_seq.
    We aim to apply the information of the macro_seq to the micro_seq.

    2 calls are made to this function:
    The 1st call uses the `query_seq` as the macro sequence, and the `shifted_segment` as the micro_seq
    The 2nd call uses the `subject_seq` as the macro sequence, and the `final_alignment` as the micro_seq

    This function implements a form of sliding window to find at least 5 consecutive matching amino acids between macro_seq and micro_seq.
    
    The outer loop (for start in range(micro_length - 4)) iterates over possible starting positions (start) in micro_seq
    where a potential match could begin. It stops 4 positions before the end of micro_seq because we need at least 5 positions 
    to check consecutive matches (range(micro_length - 4)).
    
    The inner loop (for i in range(5)) iterates 5 times, corresponding to the requirement for at least 5 consecutive matches.
    '''
    # print(f'Checking for alignment between the {direction} shifted `aligned_segment` and the `query_seq`...')
    micro_length = len(micro_seq)

    # Try to find at least 5 consecutive matching amino acids
    for start in range(micro_length - 4):
        match_count = 0
        for i in range(5):
            if micro_seq[start + i] == macro_seq[start + i]:
                match_count += 1
        if match_count >= 5:
            return True
    return False


def detect_seq_gaps(pdb_filename, cath_indices, cath_id):
    """
    Identify data entries with sequence gaps
    """
    # Step 1: Read and extract residue IDs from the PDB file
    residues = []
    
    with open(pdb_filename, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                resid = int(line[22:26].strip())
                residues.append(resid)
    
    # Step 2: Find unique and sorted residue IDs
    unique_residues = sorted(set(residues))
    
    # Step 3: Initialize a flag to track if any gaps are detected
    gap_detected = False
    
    # Step 4: Check for sequence gaps based on cath_indices
    for index_range in cath_indices:
        start, end = index_range
        
        # Collect missing residues
        missing_residues = []
        for resid in range(start, end + 1):
            if resid not in unique_residues:
                missing_residues.append(resid)
        
        # Print missing residues for the current range
        if missing_residues:
            print(f"[GAP DETECTED for cath_id {cath_id}]\nMissing residues in range {start} - {end}: {missing_residues}")
            gap_detected = True
    
    # Step 5: Return 1 if any gap is detected, otherwise return 0
    if gap_detected:
        return 1, missing_residues
    else:
        return 0, None

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

def extract_aligned_segment(query_sequence, subject_sequence, query_file, subject_file, output_file):   
    # Run blastp command
    blastp_command = f'blastp -query {query_file} -subject {subject_file} -outfmt 5 -out {output_file}'
    subprocess.run(blastp_command, shell=True, check=True)
    
    # Parse the output_file (in XML format) to extract the aligned segment
    tree = ET.parse(output_file)
    root = tree.getroot()
    
    # Find the Hsp_hseq element and extract the aligned segment
    aligned_segment = None
    for hit in root.findall('.//Hit'):
        for hsp in hit.findall('.//Hsp'):
            aligned_segment = hsp.find('Hsp_hseq').text
            break
        if aligned_segment:
            break
    
    return aligned_segment

def get_domain_sequence_from_pdb(pdb_filename, idx_range, cath_id):
    """
    The sequences come from the PDB files
    """
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
    assert len(structure) == 1

    seq = []
    start_idx, end_idx = idx_range
    idx_range_set = set(range(start_idx, end_idx + 1))  # Create a set of the range for quick lookup

    # Account for canonical amino acids, as well as UNK amino acids
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                if res_id[0] == " " and res_id[1] in idx_range_set:  # Check if it's a standard residue and within the range
                    try:
                        seq.append(protein_letters_3to1[residue.get_resname()])
                    except KeyError:
                        seq.append('U')
                elif res_id[0] != " ":
                    print('nonstandard', res_id)
    
    gap, missing_residues = detect_seq_gaps(pdb_filename, [idx_range], cath_id)

    # Add U's to the domain sequence for each gap residue
    if gap == 1 and missing_residues:
        print('\nRepairing...')
        
        # Convert the sequence to a list
        seq_list = list(seq)
        
        # Adjust indices to be relative to the start_idx
        adjusted_indices = [i - start_idx for i in missing_residues]
        
        # Insert 'U' at the specified indices
        for idx in adjusted_indices:
            seq_list.insert(idx, 'U')
        
        # Convert the list back to a string
        modified_seq = ''.join(seq_list)
        
        return modified_seq
        
    elif gap == 0:
        return ''.join(seq)

    
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
                    try:
                        seq.append(protein_letters_3to1[residue.get_resname()])
                    except KeyError:
                        seq.append('U')
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