# TACTUS - live

> Threatening activities classification toward users' security

## Useful ressources

- [Write a better commit message](https://github.com/MarcBresson/write-better-commit-messages)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 8 – Guidelines summary](https://github.com/MarcBresson/python-quick-guidelines)

## What can you use from this project

As well as the model and the entire pipeline, you could find use for the following modules:

- The skeleton class from `tactus data` allows the computation of individual keypoint speed, and more features.
- The data augmentation techniques designed for skeletons.
- The visualisation modules that takes a Skeleton object as an input.
- The pytorch MLP class from `tactus model` that has a sklearn-like signature.

## How this project has been coded

The choice of splitting the project into 3 repositories was made in the context of new-to-git developers. Indeed, by creating multiple repositories (thus multiple python packages), we limited the scope of the actions one developer could take.

### Tactus data

It contains the following:
- download and extraction of data from the UT interaction dataset.
- data augmentation techniques (flipping, scaling, sheering, rotating).
- Skeleton object that stores bounding boxes and (cleaned) keypoints. It also handles features computation.
- Yolov8 interface that takes the output of yolov8 and parses it in Skeleton objects.
- DeepSORT interface that tracks humans across a video / sequence of frame.
- RollingWindow object that stores the history of features for a given track.
- VideoCapture extension to subsample a video stream to a fix frame rate.
- a skeleton visualisation module to display bounding boxes and joints on a frame.

### Tactus model

It contains the following:
- grid search for hyperparameters tuning (hyperparameters to find the best combination of augmentations, number of features, and model hyperparameters).
- implementation of a perceptron in torch that uses the same API as sklearn.
- Classifier class that can be any sklearn classifier.
- module to evaluate AI models.
- FeatureTracker that stores RollingWindow objects in a dict, with the key being the track id.
- PredTracker that stores the prediction for each track and smooth it.

### Tactus live

It instances every objects (Yolo, DeepSORT, classifier etc.) and run the prediction on a given video stream (can be a video or an online stream).

It saves the output frames with the bouding boxes with label on them.
