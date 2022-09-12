# ADAM visual preprocessing integrations
This is the code for ADAM's visual preprocessing integrations. For the moment we have only integrated the object stroke GNN. This is based on ASU's code [here][asu_gnn].

[asu_gnn]: https://github.com/ASU-APG/adam-stage/tree/main/processing

# Setup
1. Create and activate a Python 3.9 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam-gnn python=3.9`
2. Install PyTorch and related dependencies: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
3. Install other dependencies: `pip install -r requirements.txt`
4. (Optional) If you want to run stroke extraction, install the Matlab API. (We have not yet figured out how to do this on SAGA.)

## (Optional) Matlab toolboxes
Assuming you want to run stroke extraction, be sure to install the following two Matlab toolboxes:

- Image Processing Toolkit
- Statistics and Machine Learning Toolbox

# Running
## Stroke extraction
To run stroke extraction on the M5 objects with mugs train curriculum, run:

```bash
python adam_preprocess/shape_stroke_extraction.py \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "path/to/outputs"
```

The outputs will be saved in the usual curriculum format.

Or, using the Slurm script:

```bash
sbatch extract_strokes.sh \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "path/to/outputs"
```

## Training
To train the model on say the M5 objects with mugs curriculum, evaluating on the corresponding eval curriculum:

```bash
python adam_preprocess/shape_stroke_graph_learner.py \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin"
```

Or, using the Slurm script:

```bash
cd adam_preprocess
sbatch train.sh \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin"
```

Note that neither the Python train script nor the Slurm script handles decode/inference.

## Inference/decode
To run the model trained on M5 objects with mugs curriculum, running decode for the corresponding eval curriculum:

```bash
python adam_preprocess/shape_stroke_graph_inference.py \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  --save_outputs_to "path/to/outputs"
```

To overwrite the decodes in the input curriculum files:

```bash
python adam_preprocess/shape_stroke_graph_inference.py \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  --save_outputs_to "data/curriculum/test/m5_objects_v0_with_mugs_eval"
```

To use the Slurm script:

```bash
cd adam_preprocess
sbatch predict.sh \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "path/to/outputs"
```
