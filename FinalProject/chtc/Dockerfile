# DUC v2 CHTC image.
#
# torch 2.7/cu12.8 so the image carries Blackwell kernels (sm_100/sm_120, e.g.
# CHTC's RTX PRO 6000 nodes) alongside sm_75-sm_90. A cu12.4 build tops out at
# sm_90 and crash-loops on Blackwell with "no kernel image is available".
#
# Build and push (from FinalProject/):
#   docker build -t karanvikyath17/duc-chtc:latest -f chtc/Dockerfile .
#   docker push karanvikyath17/duc-chtc:latest
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# DUC deps. munkres = Hungarian matching for clustering accuracy; the rest are
# used by custom_funcs (thrC/post_proC) and the TensorBoard writer.
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    munkres \
    tensorboard \
    pandas

# --- state-of-the-art baselines (src/sota/) ----------------------------------
# Packages the adapters import beyond the above: timm for CACTI, geomloss for
# CAGI, xgboost for RefiDiff's refinement step. Everything else they touch is
# either here already or stubbed by the adapter (IPython, matplotlib, catboost,
# mamba_ssm — replaced by a pure-PyTorch port).
RUN pip install --no-cache-dir \
    tqdm \
    timm==0.9.16 \
    geomloss==0.3.1 \
    xgboost==2.1.4

# Third-party code WITH an open-source license (DiffPuter MIT, MIRI MIT, NewImp
# and RefiDiff Apache-2.0, CACTI GPL-3.0), cloned at the commits pinned in
# src/sota/fetch.py. CAGI and AEMC-NE have no license, so they are NOT put in
# this public image; run_experiment.sh fetches them per job instead.
COPY src/sota/fetch.py /tmp/sota_fetch.py
RUN python /tmp/sota_fetch.py --root /opt/sota --licensed && rm /tmp/sota_fetch.py
ENV SOTA_ROOT=/opt/sota

WORKDIR /job
