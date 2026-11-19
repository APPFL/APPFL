:hide-toc:

.. APPFL documentation master file, created by
   sphinx-quickstart on Fri Oct 29 15:16:47 2021.
   The landing page layout is built from raw HTML blocks styled by the
   ``.appfl-landing`` rules in ``_static/css/custom.css``.

=====
APPFL
=====

.. container:: appfl-landing

   .. raw:: html

      <section class="al-hero">
        <div class="al-hero-text">
          <p class="al-hero-title">
            Privacy-preserving federated learning, <span class="al-accent">from simulation to deployment.</span>
          </p>
          <p class="al-hero-sub">
            APPFL (Advanced Privacy-Preserving Federated Learning) is an open-source framework for
            prototyping FL algorithms on a laptop, simulating them at scale on HPC clusters, and
            deploying them across real institutions, with differential privacy, secure communication,
            and plug-and-play components.
          </p>
          <div class="al-cta">
            <a class="al-btn al-btn-primary" href="tutorials/firstrun.html">Get started</a>
            <a class="al-btn al-btn-secondary" href="https://github.com/APPFL/APPFL" target="_blank" rel="noopener">
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>
              GitHub
            </a>
            <a class="al-btn al-btn-ghost" href="users/index.html">User guide &rarr;</a>
          </div>
          <div class="al-badges">
            <img src="https://img.shields.io/pypi/v/appfl?label=PyPI&amp;color=005F9E" alt="PyPI version">
            <img src="https://img.shields.io/github/stars/APPFL/APPFL?style=flat&amp;color=F7B500" alt="GitHub stars">
            <img src="https://img.shields.io/badge/python-3.8%2B-005F9E" alt="Python 3.8+">
            <img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&amp;logoColor=white" alt="Discord">
          </div>
        </div>
        <div class="al-terminal" aria-label="Quickstart commands">
          <div class="al-terminal-bar"><span></span><span></span><span></span><em>quickstart.sh</em></div>
      <pre><span class="c"># Install APPFL from source</span>
      <span class="p">$</span> git clone https://github.com/APPFL/APPFL.git
      <span class="p">$</span> cd APPFL &amp;&amp; pip install -e ".[mpi,examples]"
      <span class="p">$</span> cd examples

      <span class="c"># Simulate FedAvg with 5 clients, serially</span>
      <span class="p">$</span> python serial/run_serial.py --num_clients 5 \
          --server_config resources/configs/mnist/server_fedavg.yaml \
          --client_config resources/configs/mnist/client_1.yaml

      <span class="c"># Run asynchronous FedCompass over MPI</span>
      <span class="p">$</span> mpiexec -n 6 python mpi/run_mpi.py \
          --server_config resources/configs/mnist/server_fedcompass.yaml \
          --client_config resources/configs/mnist/client_1.yaml</pre>
        </div>
      </section>

      <section class="al-section" id="institutions">
        <p class="al-collab-title">Built together with researchers across national labs and universities</p>
        <div class="al-marquee">
          <div class="al-marquee-track">
            <div class="al-marquee-group">
              <img src="_static/anl.png" alt="Argonne National Laboratory">
              <img src="_static/uchicago.png" alt="University of Chicago">
              <img src="_static/ornl.png" alt="Oak Ridge National Laboratory">
              <img src="_static/LANL-logo.png" alt="Los Alamos National Laboratory">
              <img src="_static/LLNL-logo.png" alt="Lawrence Livermore National Laboratory">
              <img src="_static/berkeley-logo.png" alt="Lawrence Berkeley National Laboratory">
              <img src="_static/sandia.png" alt="Sandia National Laboratories">
              <img src="_static/uiuc.png" alt="University of Illinois Urbana-Champaign">
              <img src="_static/NYU-Logo.png" alt="New York University">
              <img src="_static/Iowa-State-University-Logo.png" alt="Iowa State University">
              <img src="_static/asu.png" alt="Arizona State University">
              <img src="_static/osu.svg" alt="Ohio State University">
              <img src="_static/git.png" alt="Georgia Institute of Technology">
              <img src="_static/ucsc.png" alt="UC Santa Cruz">
              <img src="_static/uc_merced.png" alt="UC Merced">
              <img src="_static/uw.png" alt="University of Washington">
              <img src="_static/cambridge.png" alt="University of Cambridge">
              <img src="_static/broad.png" alt="Broad Institute">
              <img src="_static/monai.png" alt="MONAI">
            </div>
            <!-- Duplicate group so the loop is seamless; hidden from screen readers -->
            <div class="al-marquee-group" aria-hidden="true">
              <img src="_static/anl.png" alt="">
              <img src="_static/uchicago.png" alt="">
              <img src="_static/ornl.png" alt="">
              <img src="_static/LANL-logo.png" alt="">
              <img src="_static/LLNL-logo.png" alt="">
              <img src="_static/berkeley-logo.png" alt="">
              <img src="_static/sandia.png" alt="">
              <img src="_static/uiuc.png" alt="">
              <img src="_static/NYU-Logo.png" alt="">
              <img src="_static/Iowa-State-University-Logo.png" alt="">
              <img src="_static/asu.png" alt="">
              <img src="_static/osu.svg" alt="">
              <img src="_static/git.png" alt="">
              <img src="_static/ucsc.png" alt="">
              <img src="_static/uc_merced.png" alt="">
              <img src="_static/uw.png" alt="">
              <img src="_static/cambridge.png" alt="">
              <img src="_static/broad.png" alt="">
              <img src="_static/monai.png" alt="">
            </div>
          </div>
        </div>
        <div class="al-marquee-controls">
          <button type="button" class="al-marquee-btn al-marquee-expand" aria-expanded="false"
                  onclick="var e=this.closest('section').querySelector('.al-marquee').classList.toggle('is-expanded');this.setAttribute('aria-expanded',e);this.textContent=e?'Show less':'Show all';">Show all</button>
          <button type="button" class="al-marquee-btn al-marquee-toggle" aria-label="Pause logo animation" title="Pause"
                  onclick="var p=this.closest('section').querySelector('.al-marquee').classList.toggle('is-paused');this.classList.toggle('is-paused',p);this.setAttribute('aria-label',(p?'Play':'Pause')+' logo animation');this.title=p?'Play':'Pause';">
            <svg class="al-icon-pause" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="2.5" width="3" height="11" rx="0.75"/><rect x="9.5" y="2.5" width="3" height="11" rx="0.75"/></svg>
            <svg class="al-icon-play" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 2.6v10.8a.6.6 0 0 0 .92.5l8.3-5.4a.6.6 0 0 0 0-1L5.42 2.1a.6.6 0 0 0-.92.5z"/></svg>
          </button>
        </div>
      </section>

      <section class="al-section">
        <div class="al-pillars">
          <a class="al-card al-pillar" href="tutorials/examples_scaling_test.html">
            <span class="al-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><path d="M7 7h.01M7 17h.01"/></svg></span>
            <h3>Simulate at scale</h3>
            <p>Prototype on a laptop, then simulate hundreds or thousands of clients on GPU clusters with serial, MPI, or batched-MPI runs.</p>
          </a>
          <a class="al-card al-pillar" href="users/communicator.html">
            <span class="al-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg></span>
            <h3>Deploy for real</h3>
            <p>Train across institutions over gRPC with SSL and authentication, on Globus Compute endpoints, or with Ray in the cloud.</p>
          </a>
          <a class="al-card al-pillar" href="users/technical_components.html">
            <span class="al-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><path d="M17.5 14v7M14 17.5h7"/></svg></span>
            <h3>Customize everything</h3>
            <p>Swap aggregators, schedulers, trainers, privacy mechanisms, and compressors from a YAML file, or plug in your own.</p>
          </a>
        </div>
      </section>

      <section class="al-section" id="building-blocks">
        <div class="al-section-head">
          <p class="al-eyebrow">Architecture</p>
          <h2>Six composable building blocks</h2>
          <p>A server agent and client agents are assembled from independent components. Pick one of each in your configuration, or extend the base classes.</p>
        </div>
        <div class="al-grid al-grid-3">
          <a class="al-card al-block" href="users/aggregator.html">
            <h3><span class="al-dot"></span>Aggregator</h3>
            <p>Combines one or more client models into the global model.</p>
            <div class="al-chips"><span>FedAvg</span><span>FedAdam</span><span>FedYogi</span><span>FedAsync</span><span>FedBuff</span><span>FedCompass</span><span>+ more</span></div>
          </a>
          <a class="al-card al-block" href="users/scheduler.html">
            <h3><span class="al-dot"></span>Scheduler</h3>
            <p>Decides when to aggregate as client updates arrive at different times.</p>
            <div class="al-chips"><span>Synchronous</span><span>Asynchronous</span><span>Compass</span></div>
          </a>
          <a class="al-card al-block" href="users/trainer.html">
            <h3><span class="al-dot"></span>Trainer</h3>
            <p>Runs local training and evaluation on each client's private data.</p>
            <div class="al-chips"><span>Vanilla</span><span>FedProx</span><span>ICEADMM</span><span>IIADMM</span><span>MONAI</span><span>Lightning</span></div>
          </a>
          <a class="al-card al-block" href="users/privacy.html">
            <h3><span class="al-dot"></span>Privacy</h3>
            <p>Adds protection against data leakage on the client and server sides.</p>
            <div class="al-chips"><span>Differential privacy</span><span>Opacus</span><span>Secure aggregation</span></div>
          </a>
          <a class="al-card al-block" href="users/communicator.html">
            <h3><span class="al-dot"></span>Communicator</h3>
            <p>Moves models and metadata between the server and clients.</p>
            <div class="al-chips"><span>gRPC</span><span>MPI</span><span>Globus Compute</span><span>Ray</span></div>
          </a>
          <a class="al-card al-block" href="users/compressor.html">
            <h3><span class="al-dot"></span>Compressor</h3>
            <p>Shrinks model updates with error-bounded lossy compression.</p>
            <div class="al-chips"><span>SZ2</span><span>SZ3</span><span>SZx</span><span>ZFP</span></div>
          </a>
        </div>
      </section>

      <section class="al-section" id="install">
        <div class="al-section-head">
          <p class="al-eyebrow">Install</p>
          <h2>Up and running in a minute</h2>
          <p>APPFL requires Python 3.8 or later. We recommend installing it in a fresh conda environment.</p>
        </div>
      </section>

   .. tab-set::
      :class: al-install

      .. tab-item:: pip

         .. code-block:: bash

            pip install "appfl[examples]"

      .. tab-item:: pip + MPI

         .. code-block:: bash

            pip install "appfl[mpi,examples]"

      .. tab-item:: From source

         .. code-block:: bash

            git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
            cd APPFL
            pip install -e ".[dev,mpi,examples]"

      .. tab-item:: Minimal

         .. code-block:: bash

            pip install appfl

   .. raw:: html

      <p class="al-note">Need CUDA, compressors, or help with MPI? See the <a href="install/index.html">installation guide</a>.</p>

      <section class="al-section" id="run-anywhere">
        <div class="al-section-head">
          <p class="al-eyebrow">Run anywhere</p>
          <h2>One codebase, from laptop to federation</h2>
          <p>The same server and client agents run on every backend, so moving from simulation to deployment only changes the communicator.</p>
        </div>
        <div class="al-grid al-grid-5">
          <a class="al-card al-tile" href="users/user_simulation_serial.html">
            <span class="al-tag al-tag-sim">Simulation</span>
            <h3>Serial</h3>
            <p>All clients in one process. The fastest way to debug synchronous algorithms.</p>
          </a>
          <a class="al-card al-tile" href="users/user_simulation_mpi.html">
            <span class="al-tag al-tag-sim">Simulation</span>
            <h3>MPI</h3>
            <p>One process per client (or batch of clients) on a workstation or HPC cluster.</p>
          </a>
          <a class="al-card al-tile" href="users/user_train.html">
            <span class="al-tag al-tag-dep">Deployment</span>
            <h3>gRPC</h3>
            <p>Cross-site training with SSL, token or Globus authentication, polling, and model chunking.</p>
          </a>
          <a class="al-card al-tile" href="tutorials/examples_globus_compute.html">
            <span class="al-tag al-tag-dep">Deployment</span>
            <h3>Globus Compute</h3>
            <p>Server-driven runs on HPC and cloud endpoints, with S3 or ProxyStore for large models.</p>
          </a>
          <a class="al-card al-tile" href="tutorials/examples_ray.html">
            <span class="al-tag al-tag-dep">Deployment</span>
            <h3>Ray</h3>
            <p>Spin up clusters of clients on AWS, GCP, or Azure.</p>
          </a>
        </div>
      </section>

      <section class="al-section" id="examples">
        <div class="al-section-head">
          <p class="al-eyebrow">Learn by example</p>
          <h2>Tutorials for real workloads</h2>
        </div>
        <div class="al-grid al-grid-4">
          <a class="al-card al-link" href="tutorials/firstrun.html"><h3>Quickstart</h3><p>Serial, MPI, and gRPC runs on MNIST.</p></a>
          <a class="al-card al-link" href="tutorials/examples_vit_finetuning.html"><h3>Fine-tune a ViT</h3><p>Exchange only the head layer across clients.</p></a>
          <a class="al-card al-link" href="tutorials/examples_monai.html"><h3>Medical imaging</h3><p>3D spleen segmentation with MONAI Bundles.</p></a>
          <a class="al-card al-link" href="tutorials/examples_gpuclusterrun.html"><h3>GPU clusters</h3><p>Simulations on Polaris and Delta, multi-GPU training.</p></a>
          <a class="al-card al-link" href="tutorials/examples_aws.html"><h3>Server on AWS</h3><p>Host a gRPC server on EC2 for remote clients.</p></a>
          <a class="al-card al-link" href="tutorials/examples_ssl.html"><h3>Secure gRPC</h3><p>SSL certificates and token authentication.</p></a>
          <a class="al-card al-link" href="tutorials/examples_dr_integration.html"><h3>Data readiness</h3><p>Detect and fix data issues before training.</p></a>
          <a class="al-card al-link" href="tutorials/index.html"><h3>All tutorials &rarr;</h3><p>Weights &amp; Biases, custom actions, scaling tests, and more.</p></a>
        </div>
      </section>

      <section class="al-section" id="community">
        <div class="al-section-head">
          <p class="al-eyebrow">Community</p>
          <h2>Built in the open</h2>
          <p>Questions, ideas, and contributions are welcome.</p>
        </div>
        <div class="al-grid al-grid-3">
          <a class="al-card al-social" href="https://github.com/APPFL/APPFL" target="_blank" rel="noopener">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>
            <div><h3>GitHub</h3><p>Star the repo, report issues, and open pull requests.</p></div>
          </a>
          <a class="al-card al-social" href="https://discord.gg/bBW56EYGUS" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418Z"/></svg>
            <div><h3>Discord</h3><p>Chat with the developers and other users.</p></div>
          </a>
          <a class="al-card al-social" href="https://www.youtube.com/channel/UCzwiJboiJW3dLI0UndnDy5g" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2 31.3 31.3 0 0 0 0 12a31.3 31.3 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1A31.3 31.3 0 0 0 24 12a31.3 31.3 0 0 0-.5-5.8zM9.6 15.6V8.4l6.3 3.6z"/></svg>
            <div><h3>YouTube</h3><p>Video walkthroughs of tutorials and deployments.</p></div>
          </a>
        </div>
      </section>

      <section class="al-section" id="citation">
        <div class="al-section-head">
          <p class="al-eyebrow">Citation</p>
          <h2>Cite APPFL</h2>
          <p>If you find APPFL useful for your research or development, please cite the following papers.</p>
        </div>
      </section>

   .. code-block:: latex

      @inproceedings{li2025advances,
        title={Advances in appfl: A comprehensive and extensible federated learning framework},
        author={Li, Zilinghan and He, Shilan and Yang, Ze and Ryu, Minseok and Kim, Kibaek and Madduri, Ravi},
        booktitle={2025 IEEE 25th International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
        pages={01--11},
        year={2025},
        organization={IEEE}
      }

      @inproceedings{ryu2022appfl,
         title={APPFL: open-source software framework for privacy-preserving federated learning},
         author={Ryu, Minseok and Kim, Youngdae and Kim, Kibaek and Madduri, Ravi K},
         booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
         pages={1074--1083},
         year={2022},
         organization={IEEE}
      }

   .. raw:: html

      <p class="al-ack">This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.</p>

.. toctree::
   :hidden:
   :maxdepth: 1

   install/index
   tutorials/index
   users/index
   community/index
   contribution/index
   publication/index
   changelog/index
