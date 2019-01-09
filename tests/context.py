import os
import sys

# hack / workaround / whatever so that tests can easily import main code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import degas
import degas.dataset
import degas.model
import degas.model.train
import degas.model.predict
import degas.model.helpers
