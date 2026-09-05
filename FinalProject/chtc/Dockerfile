# DUC v2 CHTC image.
#
# torch 2.7/cu12.8 so the image carries Blackwell kernels (sm_100/sm_120, e.g.
# CHTC's RTX PRO 6000 nodes) alongside sm_75-sm_90. A cu12.4 build tops out at
# sm_90 and crash-loops on Blackwell with "no kernel image is available".
#
# Build and push (from FinalProject/):
#   docker build -t <dockerhub-user>/duc-chtc:latest -f chtc/Dockerfile .
#   docker push <dockerhub-user>/duc-chtc:latest
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

WORKDIR /job
