import fiftyone as fo
import os

'https://voxel51.com/docs/fiftyone/integrations/coco.html'

dataset_exmp = fo.zoo.load_zoo_dataset("coco-2017")
dataset_exmp = fo.zoo.load_zoo_dataset(
    "coco-2017",
    label_types=["detections", "segmentations"],
    classes=["person", "car"],
    max_samples=50,
)

session = fo.launch_app(dataset_exmp)

IMAGES_DIR = os.path.dirname(dataset_exmp.first().filepath)