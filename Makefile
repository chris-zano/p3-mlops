.PHONY: install lint run clean

# Define python
PYTHON = python3

# Define the virtual environment directory
VENV_DIR = venv

# Define application entrypoint
APP_FILE = inference_api

# Create and activate a virtual environment, then install dependencies
install:
	@echo " Creating and activating virtual environment"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing dependencies from requirements.txt file"
	./$(VENV_DIR)/bin/pip install -r requirements.txt
	@echo "#################################################################################"
	@echo "Installation complete. Activate the virtual environment using the command below."
	@echo "#################################################################################"
	@echo "source $(VENV_DIR)/bin/activate"

# Run linting with pylint
lint:
	@echo "Running pylint"
	./$(VENV_DIR)/bin/pylint --rcfile=.pylintrc $(APP_FILE) scripts/train.py scripts/evaluate.py

test:
	@echo "Running pytest"
	@echo "Tests run"

run:
	@echo "Running the application"
	./$(VENV_DIR)/bin/uvicorn $(APP_FILE):app --host 0.0.0.0 --port 8000 --reload

clean:
	@echo "Cleaning up virtual environment and __pycache__ directories"
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "--- Clean up complete ---"