# FCPR: Flood Fill Correspondences for Robust Point Cloud Registration

# APPENDIX

## 1. Related Work
 
### 1.1. Traditional Registration Methods
As classical registration method, ICP [<sup>1</sup>](#refer-anchor-1) iteratively updates transformation estimation by minimizing ${l_2}$ distance between registered source points given current estimation and the reference points. Variants of ICP algorithm \cite{chen1992object,trucco1999robust,chetverikov2005robust,yang2015go,koide2021voxelized} have been proposed to resist effect of outliers and speed up convergence. NDT \cite{biber2003normal} utilizes Netwon's algorithm to maximize registered points' summed probability on source points' probability density. RANSAC \cite{fischler1981random} follows a "generation and verification" scheme, where candidate correspondences are sampled and after a certain iterations, an alignment is produced based on maximum number of consensus. Many of its variants \cite{2005Matching,5459241,2018Graph} accelerate the process and improve its robustness. FGR \cite{zhou2016fast} and Teaser \cite{yang2020teaser} manage to register points by solving global optimization problem using Geman-McClure algorithm and truncated least square algorithm respectively while at higher computation cost. A more detailed review on traditional optimization based methods can be found in \cite{yang2019performance}.


## 2. Evaluation Metrics

1.  **Relative Rotation and Translation Error**(RRE/RTE): the deviations
    from the ground-truth pose as:

    $$RRE = \arccos(\frac{trace(\hat{\textbf{R}}^{T}{\cdot}\textbf{R}) - 1}{2})$$

    $$RTE = {\lVert \textbf{t} - \hat{\textbf{t}} \rVert}_{2}$$

2.  **Registration Recall** (RR): we adopt registration recall defined
    as the fraction of the point cloud pairs whose RRE and RTE are both
    below certain thresholds: for 3DMatch&3DLoMatch benchmarks
    ${RRE < 15^{\circ}, RTE < 30cm}$, for KITTI odometry benchmark
    ${RRE < 5^{\circ}, RTE < 60cm}$.

3.  **Inlier Ratio** (IR): it is measured by the fraction of inlier
    pairs among point pairs. A pair of points are inliers if the
    distance between transformed source point under ground truth
    transformation and reference point is smaller than inlier threshold
    ${\tau_{1}}$, which is set to 0.1m for 3DMatch&3DLoMatch benchmarks.

    $$IR = \frac{1}{|\mathcal{C}|}{\sum_{(p_i^{s},p_i^{r})\in\mathcal{C}}{{\llbracket} || \textbf{R} \textbf{p}_{i}^{s} + \textbf{t} - \textbf{P}_{i}^{r}||<\tau_{1} {\rrbracket}}}$$

4.  **Feature Matching Recall** (FMR): it calculates fraction of
    putative pairs whose IR is above a certain threshold
    ${\tau_2 = 0.05}$. For the second stage of point pairs, triplets are inliers only if all three point pairs satisfy inliers
    requirement.

    $$FMR = \frac{1}{M}\sum_{i=1}^{M}{{\llbracket} IR_i > \tau_2 {\rrbracket}}$$

## 3. Implementation Details

We implement and evaluate our model with PyTorch [@paszke2019pytorch] on
hardware: CPU Intel i7-12700 and single GPU Nvidia RTX3090.

For 3DMatch&3DLoMatch benchmarks, at the first stage of triplets
initialization module, we select 10240 number of entries
${N_c = 10240}$ as number of candidate pairs and further filter via
spatial consistency to choose 512 number of pairs ${N_g = 512}$.
During flood fill process, we iterate for 120 ${\lambda_f = 120}$
number of times. We choose neighborhood size to 9 and top 1
${N_k = 1}$ corresponding pair will be added to each group of pairs.
Number of iterations ${\lambda_c}$ is set to 5. Both triangle inlier
threshold ${\sigma_d}$ and inlier threshold ${\tau_s}$ are set to 0.1m.
NMS distance threshold is set to 5cm and length threshold ${\sigma_l}$
is set to 0.05m.

For KITTI odometry benchmarks, since number of points for each frame is
much larger and each frame covers a larger area, we increase both
triangle inlier threshold ${\sigma_d}$ and inlier threshold ${\tau_s}$
to 0.6m. Length threshold ${\sigma_l}$ is also set to 0.6m. NMS distance
threshold is set to 1.2m. 24 nearest neighbor points are visited. The
rest parameter settings are the same as settings in 3DMatch&3DLoMatch
benchmarks.

We use a KPConv-FPN backbone extracted from GeoTransformer as our
backbone for feature extraction. We follow
[@huang2021predator; @qin2022geometric] to downsample the point clouds
with a voxel size of 2.5 cm on 3DMatch and 30 cm on KITTI. We keep using
4 stages KPConv-FPN layers and 5 stages KPConv-FPN layers as in
[@qin2022geometric]. Details of configuration and training parameters
can be found [@qin2022geometric].

## 4. High Inlier Ratio on Triplets Initialization

In order to generate more successfully registered initial poses, more
triplet correspondences should be inlier triplet correspondences. A
triplet correspondences are thought to be inlier triplet correspondences
when all three point pairs are inlier pairs. Based on this definition,
we make analysis on triplets initialization module and evaluate
inlier ratio of triplets. We calculate inlier ratio per scene and
visualize inlier ratio distribution through all scenes. Figure
[\[fig:triplet inlier\]](#fig:triplet inlier){reference-type="ref"
reference="fig:triplet inlier"} shows that inlier ratio has
significantly declined from 3DMatch datasets to 3DLoMatch datasets since
it is much harder to output inlier points on 3DLoMatch scenes. More
noticeable thing from both Figure
[\[fig:triplet inlier\]](#fig:triplet inlier){reference-type="ref"
reference="fig:triplet inlier"}(a) and Figure
[\[fig:triplet inlier\]](#fig:triplet inlier){reference-type="ref"
reference="fig:triplet inlier"}(b) is, inlier ratio distributions from
first stage of point pairs are very similar to the second stage of
triplets. It means for those of inlier point pairs, they basically
\"seize\" the chance to find another two point pairs that are also
inlier pairs. Numerically, we evaluate difference of *feature matching
recall* between the first stage point pairs and the second stage triplets. For 3DMatch benchmark, it slightly drops from 98.2% on first
stage pairs to 97.0% on second stage pairs and for 3DLoMatch benchmark,
it drops from 87.3% to 78.2%. This supports our proposal that our
triangle compatibility could provide good choices on two other point
pairs to form inlier triplets.




<div align=center>
<img src="https://github.com/hadleyhzy34/FCPR/blob/main/3dm_ir.png" width="360" height="270">   <img src="https://github.com/hadleyhzy34/FCPR/blob/main/3dlm_ir.png" width="360" height="270">
</div>
<div align=center>
(a) Inlier Ratio on 3DMatch                                           (b) Inlier Ratio on 3DLoMatch
</div>
Figure 1： Comparison between inlier ratio of first pairs and inlier ratio of triplets on both 3DMatch\&3DLoMatch benchmarks. We name inlier ratio of triplets as percentage of triplets that all their three pairs are inlier point pairs.





## 5. Ablation Study: CRPS Module

As the last step to refine correspondences and select pose, we enforce
another ablation analysis on our CRPS module. We set weighted SVD
[@umeyama1991least] as our baseline method. Weights are assigned based
on Equation
[\[confidence matrix\]](#confidence matrix){reference-type="ref"
reference="confidence matrix"}. Then we add LGR [@yu2021cofinet] and
RANSAC [@fischler1981random] as comparison. RANSAC is evaluated by point
correspondences ${\textbf{p}_f}$ obtained during first stage of triplets initialization and we set iteration for sampling to be 5k.

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Estimator   Module</th>
    <th class="tg-c3ow" colspan="2">RR(%)</th>
  </tr>
  <tr>
    <th class="tg-0pky">3DMatch</th>
    <th class="tg-0pky">3DLoMatch</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Weighted-SVD</td>
    <td class="tg-0pky">81.45</td>
    <td class="tg-0pky">32.57</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LGR</td>
    <td class="tg-0pky">95.75</td>
    <td class="tg-0pky">76.98</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RANSAC</td>
    <td class="tg-0pky">86.69</td>
    <td class="tg-0pky">40.60</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CRPS(ours)</td>
    <td class="tg-0pky">96.12</td>
    <td class="tg-0pky">79.06</td>
  </tr>
</tbody>
</table>

As shown in Table [1](#tab:ablation_estimator_exp){reference-type="ref"
reference="tab:ablation_estimator_exp"}, our default method with CRPS
module achieves highest registration result compared with other
approaches. Since not all initial point pairs ${p_f}$ are inlier point
pairs, non-inliers would aggregate and form group of points that are not
contributing to our final pose estimation. This explains why using SVD
weighted by corresponding pairs from all groups could not generate
highly-successful registration. RANSAC based approach does not return
accurate pose compared with LGR and our CRPS module partially due to
insufficient number of corresponding pairs to sample and increasing this
value should improve its performance. LGR model has been widely used by
[@yu2021cofinet; @bai2021pointdsc; @qin2022geometric; @chen2022sc2] and
achieved competitive registration score again when added on our module.
Nevertheless, our CRPS module still outperforms it.

# 6. Limitations

Despite competitive performance of our registration method, measurement
of feature matching matrix from first stage of our triplets
initialization module might need to be adjusted according to the
corresponding feature descriptor in order to maximize registration
performance. Any new feature descriptor may require additional work to
fine-tune the feature matching matrix to obtain higher inlier-ratio of
matches with top-k ${k=N_c}$ highest scores from feature matching
matrix. In the future, we will also work on designing triplets
initialization module that relies less on feature descriptor design.


<div id="refer-anchor-1"></div>

- [1] [P. J. Besl and N. D. McKay, "A method for registration of 3-D shapes," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2, pp. 239-256, Feb. 1992, doi: 10.1109/34.121791.](https://ieeexplore.ieee.org/document/121791)

<div id="refer-anchor-2"></div>

- [2] [Chen Y, Medioni G. Object modelling by registration of multiple range images[J]. Image and vision computing, 1992, 10(3): 145-155.](https://www.sciencedirect.com/science/article/abs/pii/026288569290066C)

<div id="refer-anchor-3"></div>

- [3] [Trucco E, Fusiello A, Roberto V. Robust motion and correspondence of noisy 3-D point sets with missing data[J]. Pattern recognition letters, 1999, 20(9): 889-898.](https://www.sciencedirect.com/science/article/abs/pii/S0167865599000550)

<div id="refer-anchor-5"></div>

- [5] []()

<div id="refer-anchor-6"></div>

- [6] []()

<div id="refer-anchor-7"></div>

- [7] []()

<div id="refer-anchor-8"></div>

- [8] []()

<div id="refer-anchor-9"></div>

- [9] []()

<div id="refer-anchor-10"></div>

- [10] []()

<div id="refer-anchor-11"></div>

- [11] []()

<div id="refer-anchor-12"></div>

- [12] []()

<div id="refer-anchor-13"></div>

- [13] []()

<div id="refer-anchor-14"></div>

- [14] []()

<div id="refer-anchor-15"></div>

- [15] []()

<div id="refer-anchor-16"></div>

- [16] []()

<div id="refer-anchor-17"></div>

- [17] []()

<div id="refer-anchor-18"></div>

- [18] []()

<div id="refer-anchor-19"></div>

- [19] []()

<div id="refer-anchor-20"></div>

- [20] []()

<div id="refer-anchor-21"></div>

- [21] []()
