import pandas as pd
import ast
import requests
import Bio
import subprocess
import xml.etree.ElementTree as ET

from Bio import SeqIO
from io import StringIO
from Bio.PDB.Polypeptide import protein_letters_3to1

def correct_alt_locs(file_path, unique_resids):
    corrected_resids = {}

    # Determine the starting residue number
    if unique_resids[0][-1].isalpha():
        current_resid_number = int(unique_resids[0][:-1])
    else:
        current_resid_number = int(unique_resids[0])

    previous = None
    for resid in unique_resids:
        current = resid 
        if current[-1].isalpha():# If current residue has an alphanumeric identifier (alt loc)
            if previous is not None and int(current[:-1]) == int(previous[:-1]): 
                # Alt Loc is continuation of prev alt loc: 34A (previous) -> 34B (current)
                corrected_resid = f"{int(corrected_resids[previous]) + 1}"
                current_resid_number = int(current[:-1]) + 2

            elif previous is not None and int(current[:-1]) == int(previous) + 1:
                # Alt Loc first appears:  53 (previous) -> 54B (current)
                corrected_resid = f"{int(previous) + 1}"
                current_resid_number = int(current[:-1]) + 1
            else:
                # Alt Loc is the first residue: None (previous) -> 34A (current)
                corrected_resid = str(current_resid_number)
                current_resid_number += 1
            
        else:
            # If current residue is not an alt loc
            if previous is not None and previous[-1].isalpha():
                # Previous residue was an Alt Loc
                if int(previous[:-1]) + 1 == int(current):
                    # 54C (previous) -> 55 (current) [NO GAP] and sequential
                    corrected_resid = str(current_resid_number)
                    current_resid_number += 1
                elif int(previous[:-1]) == int(current):
                    # 54C (previous) -> 54 (current) [NO GAP] and identical
                    corrected_resid = str(current_resid_number)
                    current_resid_number += 1
                else: 
                    # 54C (previous) -> 56 (current) [GAP] 
                    corrected_resid = current
                    current_resid_number = int(current) + 1
            elif previous is not None and previous.isdigit():
                # Previous residue was numeric resid
                if int(previous) + 1 == int(current): 
                    # 54 (previous) -> 55 (current) [NO GAP]
                    corrected_resid = str(current_resid_number)
                    current_resid_number += 1
                else:
                    # 54 (previous) -> 56 (current) [GAP]
                    corrected_resid = current # Current resid is greater than 1 away from prevous resid (GAP)
                    current_resid_number = int(current) + 1
                
            else:
                # Start of amino acid sequence (w/o) Alt Loc: 
                # None (previous) -> 12 (current)
                corrected_resid = str(current_resid_number)
                current_resid_number += 1
        
        corrected_resids[current] = corrected_resid
        previous = current

    return corrected_resids 

def renumber_pdb_file(corrected_resids, pdb_filename):
    # Read the PDB file
    with open(pdb_filename, 'r') as file:
        lines = file.readlines()
    
    # Map corrected resids back to the original lines
    new_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            
            # Extract the original residue ID from the line
            original_resid = line[22:30].strip()
            
            # Get the corrected residue ID from the dictionary
            corrected_resid = corrected_resids[original_resid]
            
            # Ensure the corrected_resid occupies 4 spaces, left justified
            corrected_resid_formatted = f"{corrected_resid:>4}"

            new_line = line[:22] + corrected_resid_formatted + ' ' + line[27:]
            new_lines.append(new_line)

    # Write the corrected PDB file back
    with open(pdb_filename, 'w') as file:
        file.writelines(new_lines)

def manual_alignment(query_seq, aligned_segment, subject_seq, output_file):
    query_length = len(query_seq)
    aligned_length = len(aligned_segment)
    subject_length = len(subject_seq)
       
    # Try shifting aligned_segment to the right
    print('Trying a right shift of the aligned segment.', file=output_file)
    for shift in range(aligned_length):
        shifted_segment = '+' * shift + aligned_segment
        if len(shifted_segment) == query_length:  # We have added enough '+' characters
            # print('`aligned_segment` length now matches `query_seq` length\n', file=output_file)
            if find_consecutive_match(query_seq, shifted_segment, output_file, direction='right'):
                print('Right shift solve was successful.\n', file=output_file)
                break
    else:
        print('Trying a left shift of the aligned segment.', file=output_file)
        # If no match found by shifting right, try shifting left
        for shift in range(aligned_length):
            shifted_segment = aligned_segment + '+' * shift
            if len(shifted_segment) == query_length: # We have added enough '+' characters
                # print('`aligned_segment` length now matches `query_seq` length\n', file=output_file)
                if find_consecutive_match(query_seq, shifted_segment, output_file,  direction='left'):
                    print('Left shift solve was successful.\n', file=output_file)
                    break

    print('A match between the `query_seq` (1) and the shifted `aligned_segment` (2) was found:', file=output_file)
    # The `shifted_sgement` contains '+' masks on either left or right sides to align with the `query_seq` because the `query_seq` may contain
    # amino acids that the BLAST+ alignment left out. Nevertheless, these amino acids are present in the pdb file and must be appended to the 'shifted_segment`
    # given the identity of those '+' amino acids in the `subject_seq`. We resolve the identity of those '+' amino acids by performing another alignment between
    # the `shifted_segment` and `subject_seq`, then replace the '+' masks based on this alignment
    print(query_seq, file=output_file) 
    print(shifted_segment, file=output_file)

    ##########################################################################################################################################################
    
    # The subject length is usually greater except in the case where residues with negative resids appear before the main sequence of interest (ex: 1o1zA00)
    if subject_length < query_length:  
         # Replace '+' with amino acids from query_seq because it's larger
        print("\n[EDGE CASE] Subject length is LESS THAN query length.", file=output_file)
        print("Replacing '+' values in the shifted `aligned_segment` with amino acids codes from the `query_seq`.\n", file=output_file)
   
        # Replace '+' with corresponding elements from query_amino_acids
        query_amino_acids = list(query_seq)
        shifted_segment_amino_acids = list(shifted_segment)
        '''
            MIVLGHRGYSAKYLENTLEAFMKAIEAGANGVELD (`subject_seq`)
        HHHHVIVLGHRGYSAKYLENTLEAFMKAIEAGANGVELD (`query_seq`)
        MIVLGHRGYSAKYLENTLEAFMKAIEAGANGVELD     (`aligned_segment`)
        ++++MIVLGHRGYSAKYLENTLEAFMKAIEAGANGVELD (`shifted_segment`), i.e. the shifted `aligned_segment`
        HHHHMIVLGHRGYSAKYLENTLEAFMKAIEAGANGVELD (`modified`)
        '''
        modified = [query_amino_acids[i] if char == '+' else char for i, char in enumerate(shifted_segment_amino_acids)]
        
    elif subject_length > query_length: # most cases
        # Find the alignment between the `shifted_segment` and the `subject_seq`
        # Typically, moving the `shifted_segment` rightwards across the `subject_seq` finds a match and allows '+' replacement
        try:  
            # print('A')
            for shift in range(subject_length):
                final_alignment = ' ' * shift + shifted_segment
                # Confirm the alignment
                if find_consecutive_match(subject_seq, final_alignment, output_file):
                    # print('B')
                    # Replace '+' with corresponding elements from subject_amino_acids
                    subject_amino_acids = list(subject_seq)
                    final_alignment_amino_acids = list(final_alignment)
                    '''
                            UUUUUUUUUUUUUUUUUUUUUUPHLSEQLCFFVQAR           (Repaired `query_seq`), i.e. GAPS filled
                    ...SLHNELKKVVAGRGAPGGTAPHVEELLPHLSEQLCFFVQARMEIAD...   (`subject_seq`)
                            VVAGRGAPGGTAPHVEELLPHLSEQLCFFVQAR              (`aligned_segment`), output from BLAST+
                            +++VVAGRGAPGGTAPHVEELLPHLSEQLCFFVQAR           (`shifted_segment`)
                            LKKVVAGRGAPGGTAPHVEELLPHLSEQLCFFVQAR           (`modified`)
                    '''
                    # Replace '+' with amino acids from subject_seq
                    print("\nReplacing '+' values in the shifted `aligned_segment` with amino acids codes from the `subject_seq`.\n", file=output_file)
                    modified = [
                        subject_amino_acids[i] if char == '+' else char
                        for i, char in enumerate(final_alignment_amino_acids)
                    ]
                    break
                    
        except IndexError:
                # print('C')
                for shift in range(subject_length):
                    # If a right shift of the `shifted_segment` across the `subject_seq` did not work, 
                    # we try shifting the `subject_seq` across the `shifted_segment`, Left & Right

                    right_shifted_subject = ' ' * shift + subject_seq # Shift the subject right by ' '
                    left_shifted_subject = subject_seq + ' ' * shift  # Shift the subject left by ' '

                    # print('D')
                    # Confirm the alignment
                    if find_consecutive_match(right_shifted_subject, shifted_segment, output_file,  code='E'):
                        # print('E')

                        shifted_segment_amino_acids = list(shifted_segment)
                        subject_amino_acids = list(right_shifted_subject)
                        '''
                        GSUUUUUUUUUUUUUUTPDYLUQLUNDKKLUSSLPNFSGIFNHLERLLDEEISRVRKDUYNDTL          (Repaired `query_seq`), i.e. GAPS filled
                           MVGEMETKEKPKPTPDYLMQLMNDKKLMSSLPNFCGIFNHLERLLDEEISRVRKDMYNDTLNGS...    (`shifted_subject`)
                        ++++++++++++++++TPDYLMQLMNDKKLMSSLPNFCGIFNHLERLLDEEISRVRKDMYNDTL          (`shifted_segment`)
                        GSMVGEMETKEKPKPTPDYLMQLMNDKKLMSSLPNFCGIFNHLERLLDEEISRVRKDMYNDTL           (`modified`)
                        '''
                        # Replace '+' with corresponding elements from subject_amino_acids
                        for i, char in enumerate(shifted_segment_amino_acids):
                            if char == '+':
                                if subject_amino_acids[i] != ' ':
                                    modified.append(subject_amino_acids[i])
                                elif query_seq[i] != 'U':
                                    modified.append(query_seq[i])
                                else:
                                    # Skip this 'i' if query_seq[i] == 'U'
                                    continue
                            else:
                                modified.append(char)
                        break
                    
                    # elif find_consecutive_match(left_shifted_subject, shifted_segment, output_file, code='G'):
                    elif find_consecutive_match(shifted_segment, left_shifted_subject, output_file, code='G'):

                        # print('G')

                        shifted_segment_amino_acids = list(shifted_segment)
                        subject_amino_acids = list(left_shifted_subject)
                        '''
                           PLTDLNQLPVQVSFEVGRQILDWHTLTSLEPGSLIDLTTPVDGEVRLLANGRLLGHGRLVEIQGRLGVRIERLTEVTISLEVUFQ	(`query_seq`)
                        ...PLTDLNQLPVQVSFEVGRQILDWHTLTSLEPGSLIDLTTPVDGEVRLLANGRLLGHGRLVEIQGRLGVRIERLTEVTIS          (`shifted_subject`)
                           PLTDLNQLPVQVSFEVGRQILDWHTLTSLEPGSLIDLTTPVDGEVRLLANGRLLGHGRLVEIQGRLGVRIERLTEVTIS		    (`aligned_segment`), output from BLAST+
                           PLTDLNQLPVQVSFEVGRQILDWHTLTSLEPGSLIDLTTPVDGEVRLLANGRLLGHGRLVEIQGRLGVRIERLTEVTIS++++++	(`shifted_segment`)
                           PLTDLNQLPVQVSFEVGRQILDWHTLTSLEPGSLIDLTTPVDGEVRLLANGRLLGHGRLVEIQGRLGVRIERLTEVTISLEVFQ     (`modified`)
                        '''
                        # Replace '+' with corresponding elements from query_amino_acids
                        for i, char in enumerate(shifted_segment_amino_acids):
                            if char == '+':
                                # modified.append(query_seq[i])
                                #if i > len(subject_amino_acids):
                                # if subject_amino_acids[i] != ' ':
                                #     modified.append(subject_amino_acids[i])
                                if query_seq[i] != 'U':
                                    modified.append(query_seq[i])
                                else:
                                    # Skip this 'i' if query_seq[i] == 'U'
                                    # It is unresolved in both the subject sequence, and the query sequence 
                                    continue
                            else:
                                modified.append(char)

                        break

                    
    ########################################################################################################################################################
    
    # Remove ' ' entries
    filtered_final_alignment = [char for char in modified if char != ' ']
    
    # Combine into a single string
    repaired_segment = ''.join(filtered_final_alignment)

    print('Successfully repaired the aligned segment with the manual solve.', file=output_file)
    print(f"Repaired aligned segment length: {len(repaired_segment)}\n", file=output_file)

    return repaired_segment

def find_consecutive_match(macro_seq, micro_seq, output_file, direction='', code=None):
    if code:
        print(code, file=output_file)
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
    # print(f'Checking for alignment between the {direction} shifted `aligned_segment` and the `query_seq`...', file=output_file)
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


def detect_seq_gaps(pdb_filename, cath_indices, cath_id, output_file):
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
            print(f"[GAP DETECTED for cath_id {cath_id}]\nMissing residues in range {start} - {end}: {missing_residues}", file=output_file)
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

def extract_aligned_segment(query_sequence, subject_sequence, query_file, subject_file, response_text):   
    # Run blastp command
    blastp_command = f'blastp -query {query_file} -subject {subject_file} -outfmt 5 -out {response_text}'
    subprocess.run(blastp_command, shell=True, check=True)
    
    # Parse the response_text (in XML format) to extract the aligned segment
    tree = ET.parse(response_text)
    root = tree.getroot()
    
    # Find the Hsp_hseq element and extract the aligned segment
    aligned_segment = None
    for hit in root.findall('.//Hit'):
        for hsp in hit.findall('.//Hsp'):
            aligned_segment = hsp.find('Hsp_hseq').text
            break
        if aligned_segment:
            break
    
    # return aligned_segment.replace("-", "") # ex: 1qouB00 contains '-' artefacts introduced by NCBI BLAST+
    return aligned_segment


def get_domain_sequence_from_pdb(pdb_filename, idx_range, cath_id, output_file):
    """
    The sequences come from the PDB files

    This function handles the prescence of UNK amino acids in PDB files and 'missing' gap residues.
    If a protein sequence has neither, the sequence given in `pdb_filename` is returned without modifications.
    This makes this function flexible to erroneous and non-erroneous provided sequences, as well as to case2 'NaN' data entries 
    with no provided sequence.
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
                    print(f'nonstandard {res_id}', file=output_file)
    
    gap, missing_residues = detect_seq_gaps(pdb_filename, [idx_range], cath_id, output_file)

    # Add U's to the domain sequence for each gap residue
    if gap == 1 and missing_residues:
        print('\nRepairing...', file=output_file)
        
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

    
def get_simple_sequence_from_pdb(pdb_filename, output_file):
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
                    print(f'nonstandard {residue.get_id()}', file=output_file)

    return ''.join(seq)

def get_uniprot_accession_from_pdb(pdb_code, duplicate_pdb_ids, output_file):  
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code.lower()}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        else:
            raise
            
    # Convert response to JSON
    pdb_data = response.json()

    try:
        # List all UnitProt IDs associated with this `pdb_code`
        uniprot_id_options = list(pdb_data[f'{pdb_code}']['UniProt'].keys())

        if pdb_code in duplicate_pdb_ids: # Your `pdb_code` reoccurs throughout the data set, and multiple subject sequences exist in UniProtKB
            print(f'Multiple subject sequences exist in UniProtKB for PDB ID [{pdb_code}]\n', file=output_file)
            return uniprot_id_options # returns a list
        else:
            # ex: PDB ID: 2a5y
            if len(uniprot_id_options) > 1:
                print(f'[EDGE CASE] PDB ID [{pdb_code}] is unique in the data set but multiple subject sequences exist in UniProtKB\n', file=output_file)
                return uniprot_id_options # returns a list
            else: # `uniprot_id_options` is a list with 1 element, meaning your unique `pdb_code` has a unique accession ID in UniProtKB
                print(f'PDB ID [{pdb_code}] is unique in the data set with NO duplicate entries\n', file=output_file)
                uniprot_id = list(pdb_data[pdb_code.lower()]['UniProt'].keys())[0] # Take the single Uniprot ID str item in the list
                return uniprot_id # returns a str
    except KeyError:
        print(f"UniProt ID not found for PDB code {pdb_code}", file=output_file)
        return None

def get_fasta_sequence_from_uniprot(uniprot_accession, output_file, cath_id='', duplicates=False):
    """
    Fetches the amino acid FASTA sequence from UniProt given a UniProt accession.
    """
    if duplicates:
        sequences = {}
        # Build `sequences` with all possible sequence matches given by the list of UniProt IDs associated with the duplicate PDB ID
        for ID in uniprot_accession: # Something like: dict_keys(['Q56222', 'Q56221', 'Q56223', 'Q56220', 'Q56219', 'Q56218', 'Q5SKZ7', 'Q56224'])
            url = f"https://www.uniprot.org/uniprot/{ID}.fasta"
            response = requests.get(url)

            if response.status_code == 200:
                fasta_content = response.text
                fasta_io = StringIO(fasta_content)
                record = SeqIO.read(fasta_io, "fasta")
                sequences[str(record.seq)] = ID # Stored the uniprot_id associated with this sequence for later access and return
            else:
                raise Exception(f"Failed to retrieve FASTA sequence for UniProt accession {ID}. Status code: {response.status_code}")

        # Open pdb file 
        pdb_filename = f"../data/{cath_id}/pdb/{cath_id}"
        # Access the sequence (`simple_sequence`) for the specific cath_id in question
        # `simple_sequence` will not fill the gap residues, but will replace UNK residues with 'U' if present in `pdb_filename`
        # This is a minimum viable solution to select a match from the possible sequences in `sequences`
        simple_sequence = get_simple_sequence_from_pdb(pdb_filename, output_file) 

        for possible_seq_match in sequences.keys():
            subject_length = len(possible_seq_match)
            query_length = len(simple_sequence)
            if query_length > subject_length:
                # A match cannot be made if the query is larger than the subject sequence. 
                # This can occur when the simple_sequence extracted from the pdb file is clearly not referring to the current UniProt sequence iteration
                continue 
                
            # Shift the `simple_sequence` rightwards across the possible_seq_match until
            for shift in range(subject_length):
                shifted_simple_sequence = ' ' * shift + simple_sequence # Right shift by x1 amino acid and check for alignment

                if len(shifted_simple_sequence) > subject_length:
                    # If we have added enough spaces to `simple_sequence` to exceed the length of the `subject_sequence`, 
                    # then no alignment was found and there is clearly not a match between `simple_sequence` and `possible_seq_match`
                    break # break and iterate to the next `possible_seq_match`
                    
                # Confirm the alignment
                if find_consecutive_match(possible_seq_match, shifted_simple_sequence, output_file):
                    # Access the uniprot_id for the matched sequence
                    uniprot_id = sequences[possible_seq_match]
                    return possible_seq_match, uniprot_id 
        return None, None
                
    else: # Simply take the str input to access the FASTA sequence
        url = f"https://www.uniprot.org/uniprot/{uniprot_accession}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            fasta_content = response.text
            fasta_io = StringIO(fasta_content)
            record = SeqIO.read(fasta_io, "fasta")
            return str(record.seq), None
        else:
            raise Exception(f"Failed to retrieve FASTA sequence for UniProt accession {uniprot_accession}. Status code: {response.status_code}", file=output_file)

def get_fasta_sequence_from_rcsb(pdb_code, output_file):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_code.lower()}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta_content = response.text
        fasta_io = StringIO(fasta_content)
        record = SeqIO.read(fasta_io, "fasta")
        return str(record.seq)
    
    except requests.exceptions.HTTPError as e:
        print(f"Failed to get FASTA sequence for PDB code {pdb_code}: {e}", file=output_file)
        return None
        
def safe_eval(val):
    """
    Function to safely convert string representation of list of tuples to actual list of tuples
    """
    if pd.isna(val):
        return val
    else:
        return ast.literal_eval(val)