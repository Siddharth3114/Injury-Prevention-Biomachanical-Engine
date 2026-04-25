import torch
from torch.utils.data import DataLoader
from stgcn_dataset import PoseDataset
from stgcn_model import Model

def verify():
    # 1. Initialize Dataset and DataLoader
    print("Loading dataset...")
    # 'pose_data.npy' is the file we generated in Phase 1
    dataset = PoseDataset(npy_dir='pose_data.npy')
    
    if len(dataset) == 0:
        print("Error: Could not find pose_data.npy. Please ensure Phase 1 was completed successfully.")
        return
        
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Simulate a batch of 2 by repeating our dummy data
    # getitem returns (4, 30, 33, 1) -> (C, T, V, M)
    sample_data = dataset[0]
    
    # We'll artificially create a batch of 2
    batch = torch.stack([sample_data, sample_data])
    print(f"Batch shape created by dataloader equivalent: {batch.shape}")
    print("Expected: (N, C, T, V, M) -> (2, 4, 30, 33, 1)")
    
    # 2. Initialize Model
    print("\nInitializing ST-GCN Model...")
    # in_channels=4 (x,y,z,visibility), num_class=0 (we aren't doing classification yet)
    model = Model(in_channels=4, num_class=0)
    
    print("Model initialized successfully.")
    
    # 3. Forward Pass
    print("\nExecuting forward pass...")
    try:
        output = model(batch)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        # Expected output shape: (N*M, Out_C, T_out, V) -> (2*1, 64, 30, 33)
        # Because stride=1, T_out = T = 30
        
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
