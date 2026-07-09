import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
import h5py
import os
from multiprocessing import Pool, cpu_count

# Paths and Configuration
training_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_training_set_full.h5"
test_file     = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_test_set_full.h5"
output_dir    = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/"
N_NEIGHBORS   = 8
BATCH_SIZE    = 5000

os.makedirs(output_dir, exist_ok=True)

# Graph Generation Function
def process_single_event(args):
    sp, label, n_neighbors = args
    if len(sp) > n_neighbors + 1:
        A_matrix = kneighbors_graph(sp, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
    elif len(sp) > 1:
        A_matrix = kneighbors_graph(sp, n_neighbors=min(n_neighbors, len(sp)-1), mode='connectivity', include_self=False).toarray()
    elif len(sp) == 1:
        A_matrix = np.array([[1]])
    else:
        return None

    edges = np.array(np.nonzero(A_matrix))
    x = torch.tensor(sp, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

# Batch Processing and Saving
def process_and_save_hdf5(hdf5_file, prefix):
    print(f"Processing {hdf5_file}...")
    
    with h5py.File(hdf5_file, 'r') as f:
        event_ids = list(f.keys())
        total_events = len(event_ids)
        
        buffer = []
        batch_counter = 0
        num_workers = max(1, cpu_count() - 2)
        pool = Pool(processes=num_workers)
        
        for i in range(0, total_events, BATCH_SIZE):
            chunk_ids = event_ids[i:i+BATCH_SIZE]
            tasks = []
            
            for eid in chunk_ids:
                sp = f[eid]['spacepoints'][:]
                label = f[eid].attrs['label']
                tasks.append((sp, label, N_NEIGHBORS))
            
            results = pool.map(process_single_event, tasks)
            valid_graphs = [g for g in results if g is not None]
            buffer.extend(valid_graphs)
            
            print(f"[{prefix}] Processed {min(i+BATCH_SIZE, total_events)}/{total_events}")
            
            chunk_output_path = os.path.join(output_dir, f"{prefix}_graphs_{batch_counter}.pt")
            torch.save(buffer, chunk_output_path)
            batch_counter += 1
            buffer = []
            
        pool.close()
        pool.join()

# Main Execution
if __name__ == '__main__':
    process_and_save_hdf5(training_file, "train")
    process_and_save_hdf5(test_file, "test")
    print("Graph generation completed.")