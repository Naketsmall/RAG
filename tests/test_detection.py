from config.configuration import TEST_IMAGE_DIR
from src.graphit import Graphit

graphit = Graphit()
results = graphit.find_objects('../' + TEST_IMAGE_DIR, debug=1)

