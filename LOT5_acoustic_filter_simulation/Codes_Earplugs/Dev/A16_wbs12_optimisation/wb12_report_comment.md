## WBS 12 Report Note

WBS 12 was originally intended as a design and optimization stage for the earplug filter, with the objective of reaching a prescribed insertion-loss target together with a controlled and as-flat-as-possible frequency response. In the broadest sense, this work package is where the validated reduced acoustic models developed in the previous WBS are meant to be used for actual design.

In practice, the work implemented here remains intentionally minimal and serves mainly as a **proof of possibility**. The script retained for this WBS, `A0_RKM_joint_optimisation.py`, performs a joint optimization of the three parameters of the equivalent film element:

1. film resistance $R$,
2. film mass $M$,
3. film stiffness $K$.

The optimization is carried out on the reduced earplug/filter model and targets an insertion loss of approximately **20 dB** under an IEC711 termination. The cost function combines three ingredients:

1. a level-matching term relative to the target IL,
2. a flatness criterion over the selected frequency band,
3. a bandwidth penalty when the response departs too strongly from the target.

The purpose of this script is not to claim a final physically identified filter design. On the contrary, the optimized values obtained for $R$, $M$, and $K$ should be understood primarily as **effective fitting parameters**. At this stage they may not correspond to a realistic film or membrane that could be manufactured directly. More reliable physical bounds and constitutive interpretations will require the measurement data planned in the next lot.

The main conclusion of this WBS is therefore methodological: the current toolbox already allows one to formulate an acoustic target and optimize a reduced filter model toward that target. Even in this simplified form, the framework is already suitable for fast exploratory studies.

This first demonstration was carried out only on a simple equivalent film element. However, the same optimization logic could later be extended to richer acoustic designs, for example:

1. Helmholtz resonators,
2. quarter-wave resonators,
3. duct section changes,
4. added porous foam elements,
5. parallel branch architectures.

Likewise, the target response does not need to remain a flat IL objective. Other target shapes could be imposed just as well, for instance:

1. lower attenuation at low frequency and stronger attenuation at high frequency,
2. band-selective attenuation,
3. broader or narrower damping plateaus,
4. load-dependent target responses.

In that sense, WBS 12 should be read as an initial optimization showcase rather than a finalized design study. It demonstrates that the reduced TMM framework is already capable of supporting target-driven filter tuning, while also making clear that the next step toward a physically credible design will depend on measurement-informed parameterization.
![img.png](img.png)