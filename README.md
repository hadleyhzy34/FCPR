# FCPR: Flood Fill Correspondences for Robust Point Cloud Registration

# APPENDIX

## 1. Related Work
 
### 1.1. Traditional Registration Methods
As classical registration method, ICP [[1]](#refer-anchor-1) iteratively updates transformation estimation by minimizing ${l_2}$ distance between registered source points given current estimation and the reference points. Variants of ICP algorithm [[2]-[6]](#refer-anchor-2) have been proposed to resist effect of outliers and speed up convergence. NDT [[7]](#refer-anchor-7) utilizes Netwon's algorithm to maximize registered points' summed probability on source points' probability density. RANSAC [[8]](#refer-anchor-8) follows a "generation and verification" scheme, where candidate correspondences are sampled and after a certain iterations, an alignment is produced based on maximum number of consensus. Many of its variants [[9]-[11]](#refer-anchor-9) accelerate the process and improve its robustness. FGR [[12]](#refer-anchor-12) and Teaser [[13]](#refer-anchor-13) manage to register points by solving global optimization problem using Geman-McClure algorithm and truncated least square algorithm respectively while at higher computation cost. A more detailed review on traditional optimization based methods can be found in [[14]](#refer-anchor-14).


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

    <div align=center>
    <img src="https://github.com/hadleyhzy34/FCPR/blob/main/assets/equation_16.png" width="300" height="56">
    </div>

4.  **Feature Matching Recall** (FMR): it calculates fraction of
    putative pairs whose IR is above a certain threshold
    ${\tau_2 = 0.05}$. For the second stage of point pairs, triplets are inliers only if all three point pairs satisfy inliers
    requirement.

    <div align=center>
    <img src="https://github.com/hadleyhzy34/FCPR/blob/main/assets/equation_17.png" width="200" height="58">
    </div>

## 3. Implementation Details

We implement and evaluate our model with PyTorch [[15]](#refer-anchor-15) on
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

For KITTI odometry benchmarks, since number of points for each frame is much larger and each frame covers a larger area, we increase both
triangle inlier threshold ${\sigma_d}$ and inlier threshold ${\tau_s}$ to 0.6m. Length threshold ${\sigma_l}$ is also set to 0.6m. NMS distance threshold is set to 1.2m. 24 nearest neighbor points are visited. The rest parameter settings are the same as settings in 3DMatch&3DLoMatch benchmarks.

We use a KPConv-FPN backbone extracted from GeoTransformer as our
backbone for feature extraction. We follow [[16]-[17]](#refer-anchor-16) to downsample the point clouds with a voxel size of 2.5 cm on 3DMatch and 30 cm on KITTI. We keep using 4 stages KPConv-FPN layers and 5 stages KPConv-FPN layers as in [[17]](#refer-anchor-17). Details of configuration and training parameters can be found [[17]](#refer-anchor-17).

## 4. High Inlier Ratio on Triplets Initialization

In order to generate more successfully registered initial poses, more
triplet correspondences should be inlier triplet correspondences. A
triplet correspondences are thought to be inlier triplet correspondences
when all three point pairs are inlier pairs. Based on this definition,
we make analysis on triplets initialization module and evaluate
inlier ratio of triplets. We calculate inlier ratio per scene and
visualize inlier ratio distribution through all scenes. Figure 1 shows that inlier ratio has
significantly declined from 3DMatch datasets to 3DLoMatch datasets since
it is much harder to output inlier points on 3DLoMatch scenes. More
noticeable thing from both Figure 1(a) and Figure 1(b) is, inlier ratio distributions from
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
<img src="https://github.com/hadleyhzy34/FCPR/blob/main/assets/3dm_ir.png" width="360" height="245">   <img src="https://github.com/hadleyhzy34/FCPR/blob/main/assets/3dlm_ir.png" width="360" height="245">

<!-- <div align=center> -->
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
(a) Inlier Ratio on 3DMatch  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    (b) Inlier Ratio on 3DLoMatch
<!-- </div> -->

Figure 1： Comparison between inlier ratio of first pairs and inlier ratio of triplets on both 3DMatch&3DLoMatch benchmarks. We name inlier ratio of triplets as percentage of triplets that all their three pairs are inlier point pairs.





## 5. Ablation Study: CRPS Module

As the last step to refine correspondences and select pose, we enforce another ablation analysis on our CRPS module. We set weighted SVD [[18]](#refer-anchor-18) as our baseline method. Weights are assigned based on Equation 4. Then we add LGR [[19]](#refer-anchor-19) and RANSAC [[8]](#refer-anchor-8) as comparison. RANSAC is evaluated by point correspondences ${\textbf{p}_f}$ obtained during first stage of triplets initialization and we set iteration for sampling to be 5k.
<br></br>
<div align=center>
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
<!-- <caption>Table 1: Evaluation Results of ablation study on Pose Estimation Strategy.</caption> -->
</table>
</div>

<div align='center'>
<caption>Table 1: Evaluation Results of ablation study on Pose Estimation Strategy.</caption>
</div>
<br/><br/>
As shown in Table 1, our default method with CRPS module achieves highest registration result compared with other approaches. Since not all initial point pairs ${p_f}$ are inlier point pairs, non-inliers would aggregate and form group of points that are not contributing to our final pose estimation. This explains why using SVD weighted by corresponding pairs from all groups could not generate highly-successful registration. RANSAC based approach does not return accurate pose compared with LGR and our CRPS module partially due to insufficient number of corresponding pairs to sample and increasing this value should improve its performance. LGR model has been widely used by [[17]](#refer-anchor-17), [[19]-[21]](#refer-anchor-19) and achieved competitive registration score again when added on our module. Nevertheless, our CRPS module still outperforms it.

# 6. Limitations

Despite competitive performance of our registration method, measurement of feature matching matrix from first stage of our triplets initialization module might need to be adjusted according to the corresponding feature descriptor in order to maximize registration performance. Any new feature descriptor may require additional work to fine-tune the feature matching matrix to obtain higher inlier-ratio of matches with top-k ${k=N_c}$ highest scores from feature matching matrix. In the future, we will also work on designing triplets initialization module that relies less on feature descriptor design.


<div id="refer-anchor-1"></div>

- [1] [P. J. Besl and N. D. McKay, "A method for registration of 3-D shapes," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2, pp. 239-256, Feb. 1992, doi: 10.1109/34.121791.](https://ieeexplore.ieee.org/document/121791)

<div id="refer-anchor-2"></div>

- [2] [Chen Y, Medioni G. Object modelling by registration of multiple range images[J]. Image and vision computing, 1992, 10(3): 145-155.](https://www.sciencedirect.com/science/article/abs/pii/026288569290066C)

<div id="refer-anchor-3"></div>

- [3] [Trucco E, Fusiello A, Roberto V. Robust motion and correspondence of noisy 3-D point sets with missing data[J]. Pattern recognition letters, 1999, 20(9): 889-898.](https://www.sciencedirect.com/science/article/abs/pii/S0167865599000550)

<div id="refer-anchor-4"></div>

- [4] [Dmitry Chetverikov, Dmitry Stepanov, Pavel Krsek, Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm, Image and Vision Computing, Volume 23, Issue 3,
2005, Pages 299-309,](https://www.sciencedirect.com/science/article/abs/pii/S0262885604001179)

<div id="refer-anchor-5"></div>

- [5] [J. Yang, H. Li, D. Campbell and Y. Jia, "Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 11, pp. 2241-2254, 1 Nov. 2016, doi: 10.1109/TPAMI.2015.2513405.](https://ieeexplore.ieee.org/document/7368945)

<div id="refer-anchor-6"></div>

- [6] [K. Koide, M. Yokozuka, S. Oishi and A. Banno, "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration," 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 11054-11059, doi: 10.1109/ICRA48506.2021.9560835.](https://ieeexplore.ieee.org/document/9560835)

<div id="refer-anchor-7"></div>

- [7] [P. Biber and W. Strasser, "The normal distributions transform: a new approach to laser scan matching," Proceedings 2003 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2003) (Cat. No.03CH37453), 2003, pp. 2743-2748 vol.3, doi: 10.1109/IROS.2003.1249285.](https://ieeexplore.ieee.org/document/1249285)

<div id="refer-anchor-8"></div>

- [8] [Fischler, Martin A. and Robert C. Bolles. “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography.” Commun. ACM 24 (1981): 381-395.](https://dl.acm.org/doi/10.1145/358669.358692)

<div id="refer-anchor-9"></div>

- [9] [O. Chum and J. Matas, "Matching with PROSAC - progressive sample consensus," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), 2005, pp. 220-226 vol. 1, doi: 10.1109/CVPR.2005.221.](https://ieeexplore.ieee.org/document/1467271)

<div id="refer-anchor-10"></div>

- [10] [Kai Ni, Hailin Jin and F. Dellaert, "GroupSAC: Efficient consensus in the presence of groupings," 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 2193-2200, doi: 10.1109/ICCV.2009.5459241.](https://ieeexplore.ieee.org/document/5459241)

<div id="refer-anchor-11"></div>

- [11] [D. Barath and J. Matas, “Graph-cut ransac,” in 2018 IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2018.](https://arxiv.org/abs/1706.00984)

<div id="refer-anchor-12"></div>

- [12] [Q.-Y. Zhou, J. Park, and V. Koltun, “Fast global registration,” in European conference on computer vision. Springer, 2016, pp. 766–782.](http://vladlen.info/papers/fast-global-registration.pdf)

<div id="refer-anchor-13"></div>

- [13] [H. Yang, J. Shi, and L. Carlone, “Teaser: Fast and certifiable point cloud registration,” IEEE Transactions on Robotics, vol. 37, no. 2, pp. 314–333, 2020.](https://arxiv.org/abs/2001.07715)

<div id="refer-anchor-14"></div>

- [14] [J. Yang, K. Xian, P. Wang, and Y. Zhang, “A performance evaluation of correspondence grouping methods for 3d rigid data matching,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 6, pp. 1859–1874, 2019.](https://arxiv.org/abs/1907.02890)

<div id="refer-anchor-15"></div>

- [15] [A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.](https://arxiv.org/abs/1912.01703)

<div id="refer-anchor-16"></div>

- [16] [S. Huang, Z. Gojcic, M. Usvyatsov, A. Wieser, and K. Schindler, “Predator: Registration of 3d point clouds with low overlap,” in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2021, pp. 4267–4276.](https://arxiv.org/abs/2011.13005)

<div id="refer-anchor-17"></div>

- [17] [ Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, and K. Xu, “Geometric transformer for fast and robust point cloud registration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 143–11 152.]()

<div id="refer-anchor-18"></div>

- [18] [S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 13, no. 04, pp. 376–380, 1991.](https://ieeexplore.ieee.org/document/88573)

<div id="refer-anchor-19"></div>

- [19] [H. Yu, F. Li, M. Saleh, B. Busam, and S. Ilic, “Cofinet: Reliable coarse-to-fine correspondences for robust pointcloud registration,” Advances in Neural Information Processing Systems, vol. 34, pp. 23 872–23 884, 2021.](https://arxiv.org/abs/2110.14076)

<div id="refer-anchor-20"></div>

- [20] [X. Bai, Z. Luo, L. Zhou, H. Chen, L. Li, Z. Hu, H. Fu, and C.-L. Tai, “Pointdsc: Robust point cloud registration using deep spatial consistency,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15 859–15 869.](https://arxiv.org/abs/2103.05465)

<div id="refer-anchor-21"></div>

- [21] [Z. Chen, K. Sun, F. Yang, and W. Tao, “Sc2-pcr: A second order spatial compatibility for efficient and robust point cloud registration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 13 221–13 231](https://arxiv.org/abs/2203.14453)
