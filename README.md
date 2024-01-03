### Dataset Preparation
In machine learning, dataset preparation involves loading, preprocessing, and augmenting data. For the solar cell defect detection task:
- **ChallengeDataset Class:** A custom class inheriting from `torch.utils.data.Dataset` is implemented, enabling efficient handling of the dataset.
- **Constructor:** Takes in a dataframe containing image paths and labels and a mode (train/validation), also initializes transformations.
- **Data Loading:** Implements `len()` to provide the dataset's length and `getitem()` to retrieve image-label pairs after applying specified transformations.
- **Transformations:** Uses `torchvision.transforms.Compose` for data transformations like converting images to tensors, normalization, and data augmentation.

### Architecture - ResNet
ResNet (Residual Network) is a deep convolutional neural network architecture known for effectively training very deep networks:
- **ResNet Model:** Implementing a variant of ResNet using sequential blocks (`ResBlock`) with skip connections.
- **ResBlock:** Consists of two sequences of Conv2D, BatchNorm, ReLU, with a skip connection connecting the input to the output after the second sequence.
- **Layer Details:** Starts with initial Conv2D, followed by blocks of varying channels and strides, ending with GlobalAvgPool, Flatten, Fully Connected (FC) layers, and a Sigmoid activation.
- **Skip Connections:** Add the input to the output within each ResBlock to tackle vanishing gradient problems.

### Training - Trainer Class
Training involves iteratively optimizing a model's parameters based on a dataset:
- **Trainer Class:** This class manages the training loop, alternating between training epochs and validation steps.
- **Early Stopping:** Incorporates an EarlyStopping criterion to halt training if the validation loss stops decreasing after a specified number of epochs.
- **Epoch-based Training:** Trains the model by iterating over the training dataset, followed by validation steps to evaluate performance.

Each aspect contributes to the overall process of building, training, and evaluating a deep learning model for solar cell defect detection.
