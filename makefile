SHELL := /bin/bash

install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black .

train:
	python sleep_pipeline.py

eval:
	echo "## Model Metrics" > report.md && \
	cat ./Results/sleep_metrics.txt >> report.md && \
	echo '\n## Confusion Matrix Plot' >> report.md && \
	echo '![Confusion Matrix](./Results/sleep_model_cm.png)' >> report.md && \
	cml comment create report.md

update-branch:
	git config --global user.name "$(USER_NAME)" && \
	git config --global user.email "$(USER_EMAIL)" && \
	git commit -am "Update with new results" && \
	git push --force origin HEAD:update
