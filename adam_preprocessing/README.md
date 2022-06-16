# ADAM visual preprocessing integrations
This is the code for ADAM's visual preprocessing integrations. For the moment we have only integrated the object stroke GNN. This is based on ASU's code [here][asu_gnn].

[asu_gnn]: https://github.com/ASU-APG/adam-stage/tree/main/processing

# Setup
1. Create and activate a Python 3.10 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam python=3.7`
2. Install PyTorch and related dependencies: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
3. Install other dependencies: `pip install -r requirements.txt`

# Running
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
