.PHONY: help install smoke cifar fedpd fedoss foogd

PY ?= python

help:
	@echo "targets: install smoke cifar fedpd fedoss foogd"
	@echo "  install  install the package in editable mode"
	@echo "  smoke    run the three CPU validation experiments"
	@echo "  cifar    run one CIFAR configuration through Docker"
	@echo "  fedpd    run the FedPD experiment launcher"
	@echo "  fedoss   run the FedOSS experiment launcher"
	@echo "  foogd    run the FOOGD experiment launcher"

install:
	$(PY) -m pip install -e .

smoke:
	$(PY) -m fedcore.experiments.exp_lemma_L
	$(PY) -m fedcore.experiments.exp_pooling_fail
	$(PY) -m fedcore.experiments.run_smoke

cifar:
	bash scripts/docker_cifar.sh

fedpd:
	bash scripts/docker_fedpd.sh

fedoss:
	bash scripts/docker_fedoss.sh

foogd:
	bash scripts/docker_foogd.sh
