# BirdWatcher

BirdWatcher is a desktop application for finding and identifying birds in video. It is an open-source hobby project focused on helping people discover which birds visit their garden.

[Download BirdWatcher](https://jesse-phitidis.github.io/BirdWatcher/)

## Repository Structure

- [`birdwatcher/`](birdwatcher/) contains the BirdWatcher desktop application, including video processing, model inference, and the user interface.
- [`training/`](training/) contains the classifier training code and configuration files.

## Technical Details

BirdWatcher uses a two-stage approach to analyse video:

1. **Bird detection:** Ultralytics [YOLO26](https://github.com/ultralytics/ultralytics) identifies bird locations in each sampled video frame.
2. **Bird classification:** Each detected bird is cropped and passed to a fine-tuned [`convnextv2_nano.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k) model from [timm](https://github.com/huggingface/pytorch-image-models), which predicts the bird species.

The classifier was fine-tuned on the [20 UK Garden Birds dataset](https://www.kaggle.com/datasets/davemahony/20-uk-garden-birds). The models used by the packaged application are exported to ONNX for runtime inference.
