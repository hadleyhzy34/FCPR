# FCPR: Flood Fill Correspondences for Robust Point Cloud Registration
icra2023

## Related Work
<!-- \label{Related Work} -->
 
### Traditional Registration Methods
As classical registration method, ICP \cite{besl1992method} iteratively updates transformation estimation by minimizing ${l_2}$ distance between registered source points given current estimation and the reference points. Variants of ICP algorithm \cite{chen1992object,trucco1999robust,chetverikov2005robust,yang2015go,koide2021voxelized} have been proposed to resist effect of outliers and speed up convergence. NDT \cite{biber2003normal} utilizes Netwon's algorithm to maximize registered points' summed probability on source points' probability density. RANSAC \cite{fischler1981random} follows a "generation and verification" scheme, where candidate correspondences are sampled and after a certain iterations, an alignment is produced based on maximum number of consensus. Many of its variants \cite{2005Matching,5459241,2018Graph} accelerate the process and improve its robustness. FGR \cite{zhou2016fast} and Teaser \cite{yang2020teaser} manage to register points by solving global optimization problem using Geman-McClure algorithm and truncated least square algorithm respectively while at higher computation cost. A more detailed review on traditional optimization based methods can be found in \cite{yang2019performance}.

