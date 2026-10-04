# Third-party release boundary

The v38 repository and release do not redistribute raw PACS, VLCS, or DomainNet images. The cached TorchVision initialization files are also excluded. Users must obtain these inputs from the respective authorized distributors and verify the registered hashes before claiming exact-identity replay.

The PACS checkpoint files are author-generated research artifacts. Their release does not grant additional rights to the underlying PACS images, ImageNet-derived initialization weights, or software dependencies. The split container-image assets preserve the exact scientific environment and remain subject to the licenses of their included operating-system, CUDA, Python, PyTorch, and supporting components.

The VLCS and DomainNet directories contain protocols, source manifests, analysis code, aggregate outcomes, and adverse primary decisions. They exclude raw images and training checkpoints. The evaluations are fixed-frame evidence and do not establish transport to an unseen source population.

See `reproduction/pacs_v3/THIRD_PARTY_NOTICES.md` and `RELEASE_ASSETS.json` for the exact asset inventory and claim boundary. This statement is a conservative research-release boundary and is not legal advice.
