# no_ball_det
Detection of No Ball events from side cam angle


1. `pip install -r requirement.txt`
2. Download the models from this [drive link](https://drive.google.com/drive/folders/1P0b5Pqgpu1xc9IxZj5D6A3uKmGYYjQea?usp=sharing) and put the model files inside `./models` directory.
3. run `python3 detect_noball_find_peak_frame_with_region3.py`

# To-Dos

- [ ] Update shoe detection dataset. Get all the previouis dataset [here](https://drive.google.com/drive/folders/19p5GvCAyA4s-xLZ5GvYcnhs2wqeXV37V?usp=sharing)
- [ ] retrain and finetune the shoe detection model
- [ ] fine sam2 or use more scalable model for faster segmentation
