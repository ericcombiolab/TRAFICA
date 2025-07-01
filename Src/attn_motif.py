import os
import numpy as np
from Bio.Seq import Seq
from Bio import motifs
from Bio.Align import PairwiseAligner
import pandas as pd
from typing import List, Tuple, Dict, Optional#, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
# import subprocess
import logomaker
import argparse


def find_high_attn_regions(
    attn_vector: np.ndarray,
    min_region_size: int = 1,
    merge_gap: int = 0,
    mean_multiplier: float = 1.0
) -> List[Tuple[int, int]]:
    """
    identify consecutive high-attn regions
    
    参数:
        attn_vector: shape (seq_len,)
        min_region_size: 
        merge_gap: merge two regions if gap < 
        mean_multiplier: default 1
        
    返回:
        高注意力区域的 (起始位置, 结束位置) 元组列表
    """
    # dynamic threshold
    threshold = attn_vector.mean() * mean_multiplier
    
    # bool mask to indicate high-attn region
    above_threshold = attn_vector >= threshold
    
    # search consecutive regions
    regions = []
    start_pos = None
    
    for pos, is_high in enumerate(above_threshold):
        if is_high and start_pos is None:
            start_pos = pos  # 区域开始
        elif not is_high and start_pos is not None:
            if pos - start_pos >= min_region_size:
                regions.append((start_pos, pos - 1))
            start_pos = None  # 重置
    
    
    if start_pos is not None and len(attn_vector) - start_pos >= min_region_size:
        regions.append((start_pos, len(attn_vector) - 1))
    
    # merge close regions (optional)
    if merge_gap > 0 and len(regions) > 1:
        merged_regions = [regions[0]]
        for current_start, current_end in regions[1:]:
            last_start, last_end = merged_regions[-1]
            
            if current_start - last_end <= merge_gap + 1:
                # merge
                merged_regions[-1] = (last_start, current_end)
            else:
                merged_regions.append((current_start, current_end))
        
        regions = merged_regions
    
    return regions



def extract_high_attn_subsequences(
    sequences: List[str],
    attn_vectors: List[np.ndarray],
    regions: List[Tuple[int, int]],
    min_high_attn_fraction: float = 0.8
) -> Dict[Tuple[int, int], Dict[str, List]]:
    """
    Extract sequences and their subsequences where attention values are above the sequence mean within each specified region.

    Args:
        sequences: List of original sequences
        attn_vectors: List of attention vectors corresponding to each sequence
        regions: List of regions to analyze (as (start, end) tuples)
        min_high_attn_fraction: Minimum fraction of high-attention positions required in a region

    Returns:
        A dictionary with structure:
        {
            (start, end): {
                'sequences': [matching original sequences],
                'subsequences': [corresponding subsequences],
                'high_attn_indices': [indices of qualifying sequences],
                'mean_attn': [mean attention for this region]
            }
        }
    """

    region_data = {}
    
    for region in regions:
        start, end = region
     
        high_attn_indices = []
        region_means = []
        
        for i, (seq, attn) in enumerate(zip(sequences, attn_vectors)):
            # obtain mean attn as threshold
            seq_mean = attn.mean()
            region_attn = attn[start:end+1]
            
            # calculate ratio of high-attn 
            high_attn_ratio = np.mean(region_attn >= seq_mean)
            
            if high_attn_ratio >= min_high_attn_fraction:
                high_attn_indices.append(i)
                region_means.append(np.mean(region_attn))
        
        # save result
        region_data[region] = {
            'sequences': [sequences[i] for i in high_attn_indices],
            'subsequences': [sequences[i][start:end+1] for i in high_attn_indices],
            'high_attn_indices': high_attn_indices,
            'mean_attn': np.mean(region_means) if region_means else 0
        }
    
    return region_data




def extract_high_attn_contiguous_segments(
    sequences: List[str],
    attn_vectors: List[np.ndarray],
    regions: List[Tuple[int, int]],
    min_contiguous_length: int = 1
) -> Dict[Tuple[int, int], Dict[str, List]]:
    """
    Extract contiguous subsequences of high-attention tokens within each specified region.

    Args:
        sequences: List of original input sequences
        attn_vectors: List of attention vectors corresponding to each sequence
        regions: List of regions to analyze as (start, end) position tuples
        min_contiguous_length: Minimum required length for contiguous subsequences

    Returns:
        A dictionary with the structure:
        {
            (region_start, region_end): {
                'sequences': [original sequences containing high-attention segments],
                'subsequences': [list of contiguous high-attention subsequences],
                'segment_positions': [(start, end) positions of each subsequence],
                'segment_means': [mean attention value for each subsequence],
                'original_seq_indices': [indices of parent sequences for each subsequence]
            }
        }
    """
    region_data = {}
    
    for region in regions:
        start, end = region
        region_length = end - start + 1
        
        all_subseqs = []
        all_positions = []
        all_means = []
        all_indices = []  # idx of original sequences
        
        for seq_idx, (seq, attn) in enumerate(zip(sequences, attn_vectors)):
            # data in current region
            region_attn = attn[start:end+1]
            region_seq = seq[start:end+1]
            seq_mean = attn.mean()  
            
            # search high-attn regions
            in_segment = False
            segment_start = 0
            current_segments = []
            
            for pos in range(region_length):
                is_high = region_attn[pos] >= seq_mean
                
                if is_high and not in_segment:
                    # fragment start
                    segment_start = pos
                    in_segment = True
                elif not is_high and in_segment:
                    # end
                    if pos - segment_start >= min_contiguous_length:
                        current_segments.append((
                            segment_start, 
                            pos - 1,
                            region_attn[segment_start:pos].mean()
                        ))
                    in_segment = False
            
            
            if in_segment and (region_length - segment_start) >= min_contiguous_length:
                current_segments.append((
                    segment_start,
                    region_length - 1,
                    region_attn[segment_start:].mean()
                ))
            
            # subsequences
            for seg_start, seg_end, seg_mean in current_segments:
                all_subseqs.append(region_seq[seg_start:seg_end+1])
                all_positions.append((start + seg_start, start + seg_end))
                all_means.append(seg_mean)
                all_indices.append(seq_idx)  
        
        # save result
        region_data[region] = {
            'sequences': sequences,  
            'subsequences': all_subseqs,
            'segment_positions': all_positions,
            'segment_means': all_means,
            'original_seq_indices': all_indices  
        }
    
    return region_data


def build_pwm(subsequences: List[str], pseudocount: float = 0.5) -> np.ndarray:
    """
    Build Position Weight Matrix (PWM) from subsequences.
    
    Args:
        subsequences: List of DNA subsequences of equal length
        pseudocount: Pseudocount value for smoothing
        
    Returns:
        PWM matrix of shape (4, seq_length) for nucleotides 'ACGT'
    """
    # Create motif object
    seq_objects = [Seq(seq) for seq in subsequences]
    motif = motifs.create(seq_objects)
    
    # Get counts and apply pseudocounts
    counts = motif.counts
    for base in "ACGT":
        for j in range(len(motif)):
            counts[base][j] += pseudocount
    
    # Normalize to probabilities
    pwm = np.zeros((4, len(motif)))
  
    for i, base in enumerate("ACGT"):
        for j in range(len(motif)):
            pwm[i, j] = counts[base][j] / (len(subsequences) + pseudocount * 4)
    
    return pwm


def load_attention_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        loaded_data = {
            'attns': data['attns'],
            'head_indices': data['head_indices'],
            'gini_scores': data['gini_scores'],
            'sequences': data['sequences'],
            'predicted_affinity':data['predicted_affinity']
        }
    return loaded_data


def PWM2TransfacFile(pwm, save_path):
    """Generate a TRANSFAC format file compliant with WebLogo requirements.
    
    Args:
        pwm: A 4xN probability matrix (row order: A, C, G, T)
        save_path: Output file path (.transfac)
    
    Notes:
        - Converts probabilities to integer counts (x100)
        - Follows strict TRANSFAC format requirements:
            * 2 spaces between keys and values
            * Tab-separated values for position data
            * UNIX line endings enforced
    """
    # Convert probabilities to integer counts (e.g., ×100)
    pwm_counts = np.round(np.array(pwm) * 100).astype(int)
    
    # TRANSFAC header (note: 2 spaces after keys)
    transfac = [
        "ID  Motif",                # 键值间2空格
        "BF  Motif",                # 键值间2空格
        "P0      A      C      G      T"  # 键值2空格 + 列间\t
    ]
    
    # Add position data (tab-aligned columns)
    for pos in range(pwm_counts.shape[1]):
        a, c, g, t = pwm_counts[:, pos]
        transfac.append(
            f"{pos+1:02d}      {a}\t{c}\t{g}\t{t}"  # 位置和值间6空格 + 列间\t
        )
    
    # File termination markers
    transfac.extend(["XX", "//"])
    
    # Write file with UNIX line endings
    with open(save_path, "w", newline="\n") as f:
        f.write("\n".join(transfac))


def align_sequences(subseqs: List[str]) -> List[str]:
    """
    Perform multiple sequence alignment using progressive alignment strategy,
    ensuring all returned sequences have equal length with gap characters.
    
    Args:
        subseqs: List of DNA sequences to be aligned
        
    Returns:
        List of aligned sequences of equal length (containing gap '-' characters)
        
    Notes:
        - Uses pairwise global alignment with affine gap penalties
        - Alignment parameters:
            * Match score: 2
            * Mismatch penalty: -1
            * Gap opening penalty: -5
            * Gap extension penalty: -0.5
        - Implements progressive alignment by iteratively adding sequences
          to a growing multiple alignment
    """
    if len(subseqs) < 2:
        return subseqs.copy()
    
    # Configure pairwise aligner with biological parameters
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    
    
    # Initialize with first sequence as reference
    reference = subseqs[0]
    aligned = [reference]
    
    for seq in subseqs[1:]:
        # Find optimal alignment to current reference
        best_alignment = None
        best_score = -float('inf')
        for alignment in aligner.align(reference, seq):
            if alignment.score > best_score:
                best_score = alignment.score
                best_alignment = alignment
        
        if best_alignment:
            # Parse alignment result
            alignment_str = str(best_alignment)
            lines = alignment_str.split('\n')
            new_ref = lines[0].replace(' ', '')  # Aligned reference
            new_seq = lines[2].replace(' ', '')  # Aligned new sequence
            
            # Calculate gap insertion positions
            gap_positions = _find_gap_positions(reference, new_ref)
            
            # Update all existing sequences with new gaps
            for i in range(len(aligned)):
                aligned[i] = _insert_gaps(aligned[i], gap_positions)
            
            # Update reference and add new sequence
            reference = new_ref
            aligned.append(new_seq)
    
    return aligned


def _find_gap_positions(old_ref: str, new_ref: str) -> List[int]:
    old = list(old_ref)
    new = list(new_ref)
    i = j = 0
    gap_positions = []
    
    while i < len(old) and j < len(new):
        if old[i] == new[j]:
            i += 1
            j += 1
        else:
            if new[j] == '-':
                gap_positions.append(j)
                j += 1
            else:
                i += 1
                j += 1
    
    while j < len(new):
        if new[j] == '-':
            gap_positions.append(j)
        j += 1
    
    return gap_positions


def _insert_gaps(seq: str, positions: List[int]) -> str:
    seq_list = list(seq)
    offset = 0
    for pos in sorted(positions):
        insert_pos = pos - offset
        if 0 <= insert_pos <= len(seq_list):
            seq_list.insert(insert_pos, '-')
            offset += 1
    return ''.join(seq_list)




def build_pwm_with_gaps(aligned_seqs: List[str], 
                       pseudocount: float = 0.1,
                       ignore_gaps: bool = True,
                       weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Build a Position Weight Matrix (PWM) from aligned sequences containing gaps (weighted version).
    
    Args:
        aligned_seqs: List of aligned sequences (may contain '-' gap characters)
        pseudocount: Pseudocount value to avoid zero probabilities (default: 0.1)
        ignore_gaps: Whether to ignore gap characters (if False, treats gaps as 5th state)
        weights: Optional weights for each sequence (must match length of aligned_seqs)
        
    Returns:
        PWM matrix (4xL or 5xL depending on ignore_gaps parameter)
        
    Raises:
        ValueError: If weights length doesn't match sequence count
        
    Notes:
        - Uses weighted counting of nucleotides at each position
        - Applies pseudocounts before normalization
        - Normalizes by total weights plus pseudocount contribution
        - Handles both DNA sequences (ACGT) and optionally gap characters
    """
    if not aligned_seqs:
        return np.zeros((4, 0))
    

    # Process weights parameter
    if weights is None:
        weights = [1.0] * len(aligned_seqs)
    elif len(weights) != len(aligned_seqs):
        raise ValueError("Weights list must match sequence count")
    
    seq_len = len(aligned_seqs[0])
    
    alphabet_size = 4 if ignore_gaps else 5
    pwm = np.zeros((alphabet_size, seq_len)) + pseudocount  # Initialize PWM with pseudocounts
    
    # Base to index mapping
    base_idx = {'A':0, 'C':1, 'G':2, 'T':3}
    if not ignore_gaps:
        base_idx['-'] = 4  # Treat gap as 5th state
    
    # Weighted counting
    for weight, seq in zip(weights, aligned_seqs):
        for pos, base in enumerate(seq):
            if base in base_idx:
                pwm[base_idx[base], pos] += weight
    
    # Column normalization (accounting for total weights + pseudocounts)
    sum_weights = sum(weights)
    pwm = pwm / (sum_weights + alphabet_size * pseudocount)
    
    return pwm



def probability_to_bits(ppm):
    """ppm to bits"""
    bits = []
    for col in ppm.T: 
        entropy = -sum(p * np.log2(p) for p in col if p > 0)
        max_bits = 2 - entropy
        bits.append([p * max_bits for p in col])
    return np.array(bits).T



def plot_attn_4_selected_heads(attn_data, save_dir, top_heads=2):
    """
    Plot attention distributions for top performing attention heads and save as PNG files.
    
    Args:
        attn_data: Dictionary containing attention data with keys:
            - 'gini_scores': Array of Gini scores for each attention head
            - 'head_indices': List of head indices
            - 'attns': 3D array of attention scores (shape: [n_layers, n_heads, seq_len])
        save_dir: Directory to save the generated plots
        top_heads: Number of top heads to plot (based on Gini scores)
        
    Returns:
        None (saves plots to specified directory)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Select top heads based on Gini scores
    idx_active_heads = np.argsort(attn_data['gini_scores'])[-top_heads:][::-1].tolist()
    
    # for i in range(len(attn_data['head_indices'])):
    for i in idx_active_heads:
        # Prepare data for plotting
        data = attn_data['attns'][:,i,:]
        positions = np.arange(data.shape[-1])  
        data_reshaped = np.vstack([positions] * data.shape[0]).T  # 复制positions以匹配data形状
        data_for_plot = np.column_stack((data_reshaped.ravel(), data.T.ravel()))

        df = pd.DataFrame(data_for_plot, columns=["Position", "Attention Score"])

        # Create plot
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=df,
            x="Position",
            y="Attention Score",
            errorbar="sd",  # Show mean +- standard deviation
            linewidth=1,
            color="blue",
        )

        # Configure plot appearance
        plt.xlim(0, len(positions) - 1)  # 如果positions是0-based，最大值是values-1
        plt.xticks(np.arange(0, len(positions), step=1))  # 每10个位置显示一个刻度

        plt.title(f"Attention distribution for head {attn_data['head_indices'][i]}")
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save plot
        plt.savefig( os.path.join(save_dir, f"attn_plot_head{attn_data['head_indices'][i]}.png"), dpi=300, bbox_inches="tight")
        plt.close()
            
            
def identify_regions(attn_data):
    avg_attn_vectors = np.mean(attn_data['attns'], axis=0)  # averaging attn by layers -> global
    all_regions = defaultdict(list)
    for idx, attn_scores in enumerate(avg_attn_vectors):
            
        regions = find_high_attn_regions(
            attn_scores,
            min_region_size=5,
            merge_gap=1,
            mean_multiplier=1.0
        )
        if len(regions) == 0:
            continue
        all_regions[ attn_data['head_indices'][idx] ]= regions # {head: high-attn regions}
    return all_regions



def motif_logomaker(pwm_path):
    with open(os.path.join(pwm_path, "motif.transfac")) as f:
        m = motifs.read(f, "transfac")

    # PFM
    pwm = np.array([m.counts[base] for base in "ACGT"])  # 获取ACGT四行数据
    # pwm_df = pd.DataFrame(pwm.T, columns=list("ACGT"))  # 转置并转为DataFrame
    # ppm = pwm / pwm.sum(axis=0) 
    column_sums = pwm.sum(axis=0)
    if (column_sums == 0).any():
        print("Warning: Some columns sum to zero, replacing with small value.")
        column_sums[column_sums == 0] = 1e-10  # or another small number
    ppm = pwm / column_sums
    
    # ppm_df = pd.DataFrame(ppm.T, columns=['A','C','G','T'])
    bits_matrix = probability_to_bits(ppm)
 
    
    # plot Logo
    plt.figure(figsize=(15, 4))

    df = pd.DataFrame(bits_matrix.T, columns=['A','C','G','T'])
    df = df + 1e-6  # Add pseudocount
    df = df.div(df.sum(axis=0), axis=1)  # Normalize columns to sum to 1
    logo = logomaker.Logo(
        df,
        color_scheme='classic',
        stack_order='big_on_top',
        fade_probabilities=True)
    
    # 
    logo.style_spines(visible=False)
    logo.ax.set_ylabel("Bits")
    logo.ax.set_xlabel("Position")
    positions = range(bits_matrix.shape[1])  # Assuming bits_matrix is position x nucleotides
    logo.ax.set_xticks(positions)  # Set ticks at every position
    
    # save
    plt.savefig(os.path.join(pwm_path, "motif.png") , dpi=300, bbox_inches='tight')
    plt.close()
                        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str, help=".")
    parser.add_argument('--attn_data_path', required=True, type=str, help=".")
    args = parser.parse_args()
    
    
    
    # # save path
    # path = f"../Motif_Attn/Base-level/PRJEB14744/50000/E2F2_TACTCA20NCG"
    # # attn data
    # attn_data = load_attention_data('../Motif_Attn/Base-level/PRJEB14744/50000/E2F2_TACTCA20NCG/Attentions/attns.npz')
    
    
    path = args.path
    attn_data = load_attention_data(args.attn_data_path)


    length_seq = len(attn_data['sequences'][0])

    
    #### attn distribution plots for active heads
    plot_attn_4_selected_heads(attn_data, save_dir=os.path.join(path, 'Attentions'), top_heads=8)


    #### Find high-attention regions   
    all_regions = identify_regions(attn_data)



    # Extract subsequences and build PWMs
    idx = 0
    region_subseqs_collect = defaultdict(dict)
    for head_idx, regions in all_regions.items():
        # 返回等长子序列，直接从原序列截取区间
        # region_subseqs =extract_high_attn_subsequences(
        #                                                 sequences=attn_data['sequences'],
        #                                                 attn_vectors=attn_data['attns'][:,idx,:],
        #                                                 regions=regions,
        #                                                 min_high_attn_fraction=0.8          
        # )
        
        ## 返回变长子序列，区间内高attn子区间
        region_subseqs = extract_high_attn_contiguous_segments(
                                                            sequences=attn_data['sequences'],
                                                            attn_vectors=attn_data['attns'][:,idx,:],  # 根据对于head的attn来进行的子序列抽取
                                                            regions=regions,  
                                                            min_contiguous_length=8
                                                        )
        
          
        idx+=1
        region_subseqs_collect[head_idx] = region_subseqs


  
    motif_path = os.path.join(path, 'Motifs')
    for head_idx, region_subseqs in region_subseqs_collect.items(): # head level
        n_regions = len(region_subseqs)
        if n_regions == 0:      # no identified high attn regions in this head
            continue 
        
        
        for region, subseqs_info in region_subseqs.items():

            subseqs = subseqs_info['subsequences']
            if len(subseqs)==0:  # no identified sub-sequences with high attn
                continue
            
            aligned_seqs = align_sequences(subseqs)     
     
            # Build PWM
            # pwm = build_pwm_with_gaps(subseqs, ignore_gaps=True)
            pwm = build_pwm_with_gaps(aligned_seqs, ignore_gaps=True)
     
     
            print(f"Region {region}: {len(subseqs)} sequences, PWM shape {pwm.shape}")

      
            motif_sub_path = os.path.join(motif_path, f'head_{head_idx}_region_{region[0]}-{region[1]}')
            if not os.path.exists(motif_sub_path):
                os.makedirs(motif_sub_path)
          
            PWM2TransfacFile(pwm, save_path=os.path.join(motif_sub_path, "motif.transfac"))
            

            # Motif logo- logomaker
            motif_logomaker(pwm_path=motif_sub_path)
     
            
    # TF-binding motif (global attention)            
    num_seqs = 100  # high-predicted affinity
    avg_attn = np.mean(attn_data['attns'],axis=1)
    region_subseqs = extract_high_attn_contiguous_segments(
                                                        sequences=attn_data['sequences'][:num_seqs],
                                                        attn_vectors=avg_attn[:num_seqs],  
                                                        regions=[(0, int(length_seq-1) )],   # -1:  for 0 position
                                                        min_contiguous_length=8
                                                    )
    pred_aff = attn_data['predicted_affinity'][:num_seqs]
    
    for region, subseqs_info in region_subseqs.items():
        
        subseqs = subseqs_info['subsequences']
        if len(subseqs)==0:  # no identified sub-sequences with high attn
            continue        
        
        
        aligned_seqs = align_sequences(subseqs)
        
        pred_aff = attn_data['predicted_affinity'][:num_seqs]
        # pwm = build_pwm_with_gaps(aligned_seqs, ignore_gaps=True, weights=pred_aff[subseqs_info['original_seq_indices']].tolist())     # weighted by predicted affinities
        pwm = build_pwm_with_gaps(aligned_seqs, ignore_gaps=True)
        
        
        print(f"Region {region}: {len(subseqs)} sequences, PWM shape {pwm.shape}")

        motif_path = os.path.join(path, 'Motifs')
        motif_sub_path = os.path.join(motif_path, f'head_all_region_{region[0]}-{region[1]}')
        if not os.path.exists(motif_sub_path):
            os.makedirs(motif_sub_path)
        
        PWM2TransfacFile(pwm, save_path=os.path.join(motif_sub_path, "motif.transfac"))
        motif_logomaker(pwm_path=motif_sub_path)
              
                   
                       
                       
                       
                        
                    
