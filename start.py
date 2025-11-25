from katabatic.models.pategan.models import PATEGAN
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from utils import discretize_preprocess

dataset_path = "raw_data/car.csv"
output_path = "discretized_data/car.csv"

discretize_preprocess(
    file_path=dataset_path,
    output_path=output_path
)

input_csv = 'discretized_data/car.csv'
output_dir = 'sample_data/car'

pipeline = TrainTestSplitPipeline(model=PATEGAN)
pipeline.run(
    input_csv=input_csv,
    output_dir=output_dir,
    synthetic_dir='synthetic/car/pategan',
    real_test_dir='sample_data/car'
)