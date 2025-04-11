# Differential Gaussian Rasterization

This rasterization engine is an altered version of the original used for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". The extensions include calculating the pixel that each Gaussian 'contributed' most to, which can be used to normalise Gaussian colours and identify surface Gaussians.

If you can make use of it in your own research, please be so kind to cite the original paper and our 3DGS-to-PC paper.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
<pre><code>@misc{stuart20253dgstopcconvert3dgaussian,
      title={3DGS-to-PC: Convert a 3D Gaussian Splatting Scene into a Dense Point Cloud or Mesh}, 
      author={Lewis A G Stuart and Michael P Pound},
      year={2025},
      eprint={2501.07478},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2501.07478}, 
}</code></pre>
  </div>
</section>