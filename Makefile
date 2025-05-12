#
# This mostly contains shortcut for multi-command steps.
#
SRC = site_forecast_app scripts tests

.PHONY: lint
lint:
	ruff $(SRC)

.PHONY: format
format:
	ruff --fix $(SRC)

.PHONY: test
test:
	pytest tests --disable-warnings
	
.PHONY: docker.build
docker.build:
	docker build -t ocf/site-forecast-app .
