# Cog-Grounding-DINO  
This is an implementation of Grounding DINO by IDEA Reseach as a [Cog](https://github.com/replicate/cog) model. Grounding DINO can detect arbitrary objects with human text inputs such as category names or referring expressions. The model architecture combines Transformer-based detector DINO with grounded pre-training to achieve open-vocabulary / text-guided object detection. See the [paper](https://arxiv.org/abs/2303.05499), [original repository](https://github.com/IDEA-Research/GroundingDINO) and this [Replicate model](https://replicate.com/alaradirik/grounding-dino).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of Grounding DINO to [Replicate](https://replicate.com).

## Basic Usage
You will need to have Cog and Docker installed on your local to run predictions. You can use Grounding DINO to query images with text descriptions of any object. To use it, simply upload an image and enter comma separated text descriptions of objects you want to query the image for. Expected input arguments are:  

- **image:** your input image
- **query:** text queries describing objects you want to detect, separate queries with commas
- **box_threshold:** chooses the boxes whose highest similarities are higher than a box_threshold
- **text_threshold:** extracts the words whose similarities are higher than the text_threshold as predicted labels


To run a prediction:
```bash
cog predict -i image=@mugs.png -i query="a pink mug" -i box_threshold=0.2 -i text_threshold=0.25 
```

To build the cog image and launch the API on your local:
```bash
cog run -p 5000 python -m cog.server.http
```

## References 
```
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```