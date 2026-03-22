.PHONY: install generate train evaluate benchmark export visualize test lint notebook clean

install:
	pip install -r requirements.txt
	pip install -e .

generate:
	python -c "from simulation.trajectory_gen import TrajectoryGenerator; gen = TrajectoryGenerator(); gen.generate_batch(100)"

train:
	python -m training.train_hgv

evaluate:
	python -m evaluation.evaluate

benchmark:
	python -m evaluation.benchmark

export:
	python -m deployment.tensorrt_export

visualize:
	python -m visualization.trajectory_viz

test:
	pytest tests/ -v

lint:
	black --check . --line-length 100
	flake8 . --max-line-length=100

notebook:
	jupyter notebook

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf *.egg-info
