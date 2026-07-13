A microbubble tracking system for use in super-resolution ultrasound. Christened **TrajMB-ODE**, the proposed architecture entails a a continuous latent-ODE driven deterministic encoder-decoder framework that learns microbubble dynamics and generalizes for complex motion scenarios.

The contributions of the proposed architecture are three-fold. 
1. An encoder architecture that leverages a derivative-informed Spatio-Temporal (ST) gate to account for spatio-temporal reliability, acting as an attentional filter. This design penalizes physiologically unrealistic motion.
2. Designing the system such that it is able to model microbubble trajectories as continuous-time solutions to an ODE enables the proposed architecture to learn the underlying physics of blood flow implicitly. This implementation can be considered an improvement that provides the framework with inherent resilience to noise and temporal discontinuities.
3. The model is validated on a high-fidelity simulation framework using the modified nonlinear Piepenbrock motion model proposed by Zhang et al. (Zhang Y, Zhou W, Huang L, Shao Y, Luo A, Luo J, Peng B. Efficient Microbubble Trajectory Tracking in Ultrasound Localization Microscopy Using a Gated Recurrent Unit-Based Multitasking Temporal Neural Network. IEEE Trans Ultrason Ferroelectr Freq Control. 2024 Dec;71(12: Breaking the Resolution Barrier in Ultrasound):1714-1734. doi: 10.1109/TUFFC.2024.3424955. Epub 2025 Jan 8. PMID: 38976462.) and the ULTRA-SR Challenge Dataset (M Lerendegui, K Riemer, B Wang, C Dunsby, M-X Tang, BUbble Flow Field: a Simulation Framework for Evaluating Ultrasound Localization Microscopy Algorithms, ArXiv, 1 Nov 2022)

To generate the simulated data with linear and nonlinear microbubble trajectories spaced uniformly (equal time spacing between microbubble points), specify the scenario and use
```bash
py generate_tracks.py -s 2
```
To generate the simulated data with linear and nonlinear microbubble trajectories spaced non-uniformly (unequal time spacing between microbubble points), specify the scenario and use
```bash
py generate_tracks_irregular.py -s 2
```
The generated tracks can be visualized using the snippet
```python
data= load('data/nonlinear/scenario_2/train.npy',allow_pickle=True).item()
#the loaded data is a dictionary with each key corresponding to the length of the microbubble track and every value a tensor of size number_of_trajectories x track length x 2 (coordinates - [x,z] in the third dimension)
plt.plot(data['5'][0,:,0],data['5'][0,:,1],'r-o')
plt.show()
```

To train the model, the `torchdiffeq` module will be required. Do
```python
pip install torchdiffeq
```
The model can be evaluated on *in-vivo* and *in-vitro* data using the `evalTrajODE` Jupyter notebook. 
