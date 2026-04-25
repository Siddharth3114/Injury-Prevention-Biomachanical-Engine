import numpy as np

# Shape: (4, 30, 33, 1) -> (C=4, T=30, V=33, M=1)
# Create 30 frames
T = 30
data = np.zeros((4, T, 33, 1))

# Default standing pose (rough estimates)
# Hip: L=23, R=24
# Knee: L=25, R=26
# Ankle: L=27, R=28

base_y = 0.5
base_x = 0.5

for t in range(T):
    # Simulate a jump landing (downward motion starts at t=10)
    # y increases downwards
    jump_y_offset = max(0, (t - 10) * 0.01) if t > 10 else 0
    hip_y = 0.5 + jump_y_offset
    
    # Hips
    data[:2, t, 23, 0] = [base_x + 0.1, hip_y] # L hip
    data[:2, t, 24, 0] = [base_x - 0.1, hip_y] # R hip
    
    # Valgus collapse: left knee moves inward (towards x=0.5) significantly during jump
    knee_inward = max(0, (t - 15) * 0.01) if t > 15 else 0
    data[:2, t, 25, 0] = [base_x + 0.1 - knee_inward, hip_y + 0.2] # L knee shifts left (inward)
    data[:2, t, 26, 0] = [base_x - 0.1, hip_y + 0.2] # R knee stable
    
    # Ankles stable
    data[:2, t, 27, 0] = [base_x + 0.1, 0.9] # L ankle
    data[:2, t, 28, 0] = [base_x - 0.1, 0.9] # R ankle
    
    # Set Z to zero for 2D-like projection in 3D
    data[2, t, :, 0] = 0
    # Visibility to 1
    data[3, t, :, 0] = 1.0

np.save("pose_data_synthetic.npy", data)
print("Saved pose_data_synthetic.npy")
