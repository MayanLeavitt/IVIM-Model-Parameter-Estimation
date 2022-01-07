


<h1 align="center"> Improved Architecture for MRI IVIM Model Prediction </h1> 
<h3 align="center"> Project A 044167 – Spring 2021 </h3>
<h5 align="center"> Technion – Israel Institute of Technology </h5>

  <p align="center">
    <a href="https://github.com/MayanLeavitt"> Mayan Leavitt </a> •
    <a href="https://github.com/idankinderman"> Edan Kinderman </a> 
  </p>
  
<br />
<br />

<p align="center">
  <img src="https://user-images.githubusercontent.com/62880315/143781288-85528348-b0f7-4291-839f-d5451dcfc256.gif" alt="animated" />
  <img src="https://user-images.githubusercontent.com/62880315/143781388-e9498d2d-c151-4462-872d-8ce34bf38912.gif" alt="animated" />
  <img src="https://user-images.githubusercontent.com/62880315/143781276-84e58e17-62b3-495a-9421-c3643a0129ff.gif" alt="animated" />
</p>

<br />
<br />

- [Summary](#summary)
- [DW MRI](#dw-mri)
- [The IVIM model](#the-ivim-model)
- [Existing solvers](#existing-solvers)
- [The proposed nets](#the-proposed-nets)
- [Results](#results)
- [Clinical Data](#clinical-data)
- [Files and Usage](#files-and-usage)
- [References and credits](#references-and-credits)


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="summary"> :book: Summary </h2>

We proposed a novel architecture and training protocol for a network, which estimate the IVIM model parameters from a DW MRI signal. By estimating these parameters, it is possible to distinguish between healthy and pathological tissues. we implemented it and tested the proposed network on simulated data as well as on clinical data. 
Our network achieved significantly improved results in terms of accuracy (RMSE). We conclude that the proposed network is more robust to different image quality conditions and thus can deal with noised data in a manner that accommodates real life clinical settings. 

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="dw-mri"> :brain: DW MRI </h2>

Diffusion-Weighted (DW) MRI is an imaging technique which deploys the diffusion process of molecules in order to extract diffusion bio-markers [[1]](#ref1).
By using signals with different b values (proportional to the gradient of the magnetic field inside the scanner) the motion sensitivity is varied.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="the-ivim-model"> :open_book: The IVIM model</h2>

The Intravoxel Incoherent Motion (IVIM) model proposes a relation between the obtained signal and the tissue parameters in each voxel [[1]](#ref1).
By estimating these parameters, it is possible to distinguish between healthy and pathological tissues.

<br />

Below is the IVIM model:


<p align="center">
<h3 align="center"> S<sub>n</sub> = S<sub>0</sub> (F &middot; e<sup>-b<sub>n</sub>&middot;D<sub>p</sub></sup> + (1-F) &middot; e<sup>-b<sub>n</sub>&middot;D</sup>) </h3>
</p>

<br />

The known paramters are:
* S<sub>n</sub> - the signal value that was obtained with b<sub>n</sub> (the solvers input).
* b<sub>n</sub> - the n-th b-value.
* S<sub>0</sub> - the signal value with b=0.

The bio-markers that are estimated:
* F - the perfusion fraction.
* D<sub>p</sub> - the pseudo-diffusion coefficient.
* D - the diffusion coefficient.

Following are parametric maps of an osteosarcoma tumor, obtained using classic algorithms [[2]](#ref2):

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/82229571/143095269-1f1a4bcd-b778-4f97-b859-9bd1d4b04872.png" align="center" alt="Parameters maps" width="400" height="150">
</p>

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="existing-solvers"> :mortar_board: Existing solvers</h2>

We implemented two types of solvers based on existing solutions to the IVIM estimation problem (see references).

<br />

The first solver is an LSQ-based method (the "Classic solver") [[3]](#ref3). It adjusts the parameter values of the IVIM model by minimizing the MSE between the
fitted signal and the acquired signal. Below is an outline of the classic solver flow:

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/143688243-8e3ee79e-5528-41b0-9e9f-67e45d8273bb.PNG" align="center" alt="Parameters maps" width="350" height="350">
</p>

<br />

The second solver is a DNN with the next properties [[4]](#ref4):
* The input vector is the normalized signal for each b-value.
* The output vector is the parameter predictions – F, D, D<sub>p</sub>.
* The network consists of 3 fully connected hidden layers.
* ELU activation.
* MSE loss function calculated on the signal value.
* Training is done with 100,000 samples with an SNR of 60.
* Adam Optimizer.
* Early stopping – after 10 unimproved epochs.
* The estimation is done voxel wise.

Below is a schematic of the net:

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/143688186-859181cf-7654-4b58-9fbf-20275ea04d32.PNG" align="center" alt="Parameters maps" width="600" height="275">
</p>

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="the-proposed-nets"> :thought_balloon: The proposed nets</h2>

Differnet clinical settings lead to aquired signals with varying SNR's, but the solvers described above don't deal well with noised data.
* Classic solvers are sensitive to noise and can’t utilize the SNR.
* The DNN needs to be trained in every new clinical setting.

As a sulotion, we propose a modified network architecture and training protocol in order to achieve improved robustness to varying image quality conditions (i.e. unknown SNR).
We examined two new networks:
1. A net with the same architecture as the “basic net”, but with richer training data - using 16 sets of samples, such that each set has a different SNR.
2. A net that is similar to the previous net but with an addition of the SNR as an input. This way the net has more information about the noise it needs to deal with.

Both networks train with 1,000,000 samples.

These solutions are based on the idea of defending against adversarial attacks [[5]](#ref5).

Below is an illustration of the training data:

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/143688206-bdc4e309-40d5-46a4-bf0a-77f0fae7f509.PNG" align="center" alt="Parameters maps" width="650" height="230">
</p>

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="results"> :bar_chart: Results</h2>

The comparison between the different solvers is presented using two measures:
* For the parameters - using RMSE.
* Calculating the accumulated loss - using MSE between the IVIM model signals.

We devided each comparison in to two - first, comapring the basic net and the classic solver, and then comparing the three nets.


<h4> D comaprison </h4>
<img src="https://user-images.githubusercontent.com/82229571/146633883-8635777c-cdde-4b8f-a18b-93ba31525630.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/82229571/146633892-1f23cd6d-3360-49bc-b544-d10bfcef05e3.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />


<h4> D<sub>p</sub> comaprison </h4>
<img src="https://user-images.githubusercontent.com/82229571/146633886-b1e6f56c-1df8-4786-9f34-52c561698c24.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/82229571/146633891-023f4f0c-b051-4358-98c3-583657c17935.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<h4> F<sub>p</sub> comaprison </h4>
<img src="https://user-images.githubusercontent.com/82229571/146633884-43869eff-c129-4782-874d-62d6e5fe3054.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/82229571/146633894-0abfd40e-93d7-460c-88a8-f4db8cd91d70.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<h4> Loss comaprison </h4>
<img src="https://user-images.githubusercontent.com/82229571/146633885-a520b89f-3a69-4635-b190-608cb678cd1d.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/82229571/146633895-e5490197-8d69-450a-9c3e-526f90db387a.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="clinical-data"> :stethoscope: Clinical Data</h2>

Below are examples of bio-marker maps that were created from the solvers ouputs, for clinical data inputs (upper abdomen DWI MRI scans).
It is important to notice that clinical data always contains a significant amount of noise or inaccuracies.
In addition, there are less b-values in each input. This explains the black "stains" in the Classic Solver's maps - there aren't enough b-values under 200 and the solver fails to converge in these areas.
The nets, however, are capable of estimating the bio-markers even in ares with significant noise.

<br />
<br />

<img src="https://user-images.githubusercontent.com/62880315/147814097-5dfbdc00-0b29-4f02-be34-d597a8940e42.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/62880315/147814107-9be6928c-9671-486d-a59f-1598d00803c8.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<img src="https://user-images.githubusercontent.com/62880315/147814115-32687f27-6c1b-4b2d-8a88-9d689c1b3f39.png" align="left" width="385" height="280">
<img src="https://user-images.githubusercontent.com/62880315/147814125-c2adfa3d-8f2d-4a32-8489-89dded967484.png" align="right" width="385" height="280">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<br />
<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="files-and-usage"> :man_technologist: Files and Usage</h2>

| File Name        | Description           |
| ---------------- |:-----------------:|
| top_defenitions.py | Defines the constant variables, and the functions for creating a net |
| least_Square_Solver.py | Implementation of the classic solver |
| solver_main.py | Executes the classic solver |
| net_training_and_testing.py | Implementation of the nets training and testing protocols |
| basic_net_defenition.py | Creates the training data for the basic net and trains it |
| different_vars_net_defenition.py | Creates the training data for the first proposed net and trains it |
| vars_with_input_net_defenition.py | Creates the training data for the second proposed net and trains it |
| compare_solvers.py | Plots the comparison graphs between the different solvers |
| data_simulations.py  | Creates data that is used to test the solvers |
| clinical_data_test.py | Tests all four solvers on clinical data |
| weights | Contains the net weights |
| classic_solver_estimations | Contains the classic solver's preformance variables |

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* Project supervisor: Shira Nemorovsky-Rotman. Some of the algorithms were implemented based on her code.
* The clinical data belongs to Harvard Medical School.
* <a id="ref1">[[1]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.995.1925&rep=rep1&type=pdf)</a> The IVIM model and DW MRI are explained in "Diffusion-Weighted MRI in the Body: Applications and Challenges in Oncology", D.-M. K. and D. J. Collins, American Journal of Roentgenology, pp. 1622-1635, 2007. [↩](#dw-mri)
* <a id="ref2">[[2]](https://www.sciencedirect.com/science/article/abs/pii/S1361841512001703)</a> The parametric maps of an osteosarcoma tumor were taken from "Reliable estimation of incoherent motion parametric maps from diffusion-weighted MRI using fusion bootstrap moves", M. F., J. M. Perez-Rossello, M. J. Callahan, S. D. Voss, K. E., R. V. Mulkern and S. K. Warfield,  Medical image analysis, pp. 325-336, 2013. [↩](#the-ivim-model)
* <a id="ref3">[[3]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0211911)</a> The classic solver is proposed in “Estimation of intravoxel incoherent motion parameters using low b-values”, C. Ye, D. Xu, Y. Qin, L. Wang, R. Wang, W. Li, Z. Kuai, Y. Zhu, PloS one, 2019. [↩](#existing-solvers)
* <a id="ref4">[[4]](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.27910)</a> The Basic net architecture is presented in "Deep learning how to fit an intravoxel incoherent motion model to diffusion‐weighted MRI", S. Barbieri, O. J. Gurney‐Champion, R. K. and H. C. Thoeny,  Magnetic resonance in medicine, pp. 312-321, 2020. [↩](#existing-solvers)
* <a id="ref5">[[5]](https://arxiv.org/abs/1412.6572)</a> Adversarial noise is examined in "Explaining and harnessing adversarial exmaples", I. J. Goodfellow, J. Shlens and C. Szegedy, arXiv preprint, 2014. [↩](#the-proposed-nets)
