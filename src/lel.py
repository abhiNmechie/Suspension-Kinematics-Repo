import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. YOUR DATA
# =============================================================================

# Paste your 25 values from your Python Solver here
your_caster_deg = np.array([4.313993251772856,
4.338837661476795,
4.363686586350321,
4.388540055448587,
4.413398098456041,
4.438260745641002,
4.463128027884016,
4.487999976687319,
4.512876624184566,
4.537758003150774,
4.562644147012573,
4.587535089858708,
4.612430866450806,
4.637331512234432,
4.662237063350434,
4.687147556646634,
4.712063029689644,
4.736983520777274,
4.761909068951013,
4.78683971400892,
4.811775496518928,
4.836716457832337,
4.861662640097808,
4.886614086275571,
4.911570840152159
    # ... paste your 25 values here ...
])

# Automatically generates an array from 0 to 25 mm with the exact same length (25)
your_bump_travel = np.linspace(1, 25, len(your_caster_deg)) 


# Paste the 25 Lotus values here
lotus_caster_deg = np.array([
    4.9068, 4.8821, 4.8574, 4.8327, 4.8080, 4.7833, 4.7586, 4.7339, 4.7091, 
    4.6844, 4.6597, 4.6350, 4.6103, 4.5856, 4.5609, 4.5362, 4.5115, 4.4868, 
    4.4621, 4.4374, 4.4127, 4.3880, 4.3633, 4.3386, 4.3139
])

# Automatically generates an array from 25 to 0 mm with the exact same length (25)
lotus_bump_travel = np.linspace(25, 1, len(lotus_caster_deg))


# =============================================================================
# 2. PLOTTING SIDE-BY-SIDE IN A NEW WINDOW
# =============================================================================
# (The rest of the plotting code remains exactly the same)

plt.ion()  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# --- Left Subplot: Your Code Results ---
ax1.plot(your_bump_travel, your_caster_deg, marker=' ', color='blue', linewidth=2, label=' ')
ax1.set_title("mein code!!")
ax1.set_xlabel("Bump")
ax1.set_ylabel("Caster")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Right Subplot: Lotus Shark Results ---
ax2.plot(lotus_bump_travel, lotus_caster_deg, marker='s', color='orange', linewidth=2, label='Lotus Shark')
ax2.set_title("Lotus")
ax2.set_xlabel("Bump")
ax2.set_ylabel("Caster")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show(block=True)