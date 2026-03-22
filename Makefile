# PhysNet-HGV Makefile

.PHONY: install train test visualize lint export clean

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python physnet-hgv/training/train_hgv.py

test:
	pytest physnet-hgv/tests/ -v

visualize:
	python physnet-hgv/visualization/trajectory_viz.py

lint:
	black .
	flake8 .

export:
	python physnet-hgv/deployment/tensorrt_export.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
