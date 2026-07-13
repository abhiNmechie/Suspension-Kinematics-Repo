import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bumpmetrics import camber_front,camber_rear,caster_front,caster_rear,toe_front,toe_rear,kpi_front,kpi_rear,rc_arr

bump_travel=np.linspace(-25,25,51)
fig, axs=plt.subplots(2,3,figsize=(15,8))
ax_list=axs.flatten()

ax_list[0].plot(bump_travel,camber_front)
ax_list[0].set_title("Front Camber")
ax_list[0].set_xlabel("Bump Travel (mm)")
ax_list[0].set_ylabel("Camber (deg)")
ax_list[0].grid(True)

ax_list[1].plot(bump_travel,caster_front)
ax_list[1].set_title("Front Caster")
ax_list[1].set_xlabel("Bump Travel (mm)")
ax_list[1].set_ylabel("Caster (deg)")
ax_list[1].grid(True)

ax_list[2].plot(bump_travel,toe_front)
ax_list[2].set_title("Front Toe")
ax_list[2].set_xlabel("Bump Travel (mm)")
ax_list[2].set_ylabel("Toe (SAE, deg)")
ax_list[2].grid(True)

ax_list[3].plot(bump_travel,kpi_front)
ax_list[3].set_title("Front KPI")
ax_list[3].set_xlabel("Bump Travel (mm)")
ax_list[3].set_ylabel("kpi (deg)")
ax_list[3].grid(True)

ax_list[4].plot(bump_travel,rc_arr)
ax_list[4].set_title("RC height from grnd")
ax_list[4].set_xlabel("Bump Travel (mm)")
ax_list[4].set_ylabel("RC_Z (mm)")
ax_list[4].grid(True)

ax_list[5].axis('off')

plt.tight_layout() 
plt.show()
