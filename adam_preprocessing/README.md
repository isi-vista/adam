# ADAM visual preprocessing integrations

This is the code for ADAM's visual preprocessing integrations. For the moment we have only integrated the object stroke GNN. This is based on ASU's code [here][asu_gnn].

[asu_gnn]: https://github.com/ASU-APG/adam-stage/tree/main/processing

# Setup

1. Create and activate a Python 3.9 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam-gnn python=3.9`
2. Install PyTorch and related dependencies: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
3. Install other dependencies: `pip install -r requirements.txt`
4. (Optional) If you want to run stroke extraction, install the Matlab API. On SAGA you can install this using: `cd /nas/gaia/adam/matlab/extern/engines/python && pip install .`.

## (Optional) Matlab toolboxes

Assuming you want to run stroke extraction, be sure to install the following two Matlab toolboxes:

- Image Processing Toolkit
- Statistics and Machine Learning Toolbox

## (Optional) Color segmentation

If you want to run color segmentation, you'll also need to download the Matlab code for color segmentation. This code is stored in a separate repo ([adam_MCL_CCP][adam_MCL_CCP]) due to licensing issues. To install it, download from https://github.com/isi-vista/adam_MCL_CCP/archive/refs/heads/main.zip. Extract the zip file somewhere, then move the Matlab code contents of the extracted directory into this directory. More precisely, in Bash:

```bash
cd /path/to/adam_preprocessing
ZIP_NAME=adam_MCL_CCP-main
if [[ ! -d tmp ]]; then
    unzip "$ZIP_NAME".zip
    # We only need to move the CCP_seg.m code and its dependencies
    mv "$ZIP_NAME"/{CCP_seg.m,ColorPalette,MeanShift,PeterKovesi,StructuredEdgeDetection,piotr_toolbox,others} .
    rmdir "$ZIP_NAME"
fi
```

# A note about Matlab on SAGA

Matlab on the cluster is subject to licensing restrictions. Specifically, if you want to run it on a SAGA host you must first personally activate Matlab using your license on that host. There [seems to be][matlab_lim] a two-host activation limit. Between these two problems, it's not practical to run Matlab code on `scavenge` or `ephemeral`. Keep this in mind.

It's easiest to activate Matlab from inside a [VNC session][vnc]. Once inside, you'll want to run a terminal emulator and from there run `/nas/gaia/adam/matlab/bin/activate_matlab.sh`.

Alternatively, you can try to activate Matlab through the [Licensing Center][lic_cent] on the Matlab website. Click on the row for your license, go to the tab "Install and activate," then choose the link "View Current Activations" and click on the button "Activate a computer." For the host ID, enter the output of the following Bash function on the host in question:

```bash
function getHostID() {
    /sbin/ifconfig |
      perl -ne 'print if not /docker|virbr/../^$/' |
      grep -F 'ether' |
      tail -n 1 |
      sed -e 's/ \+ether \(\([0-9a-f]\{2\}:\)\{5\}[0-9a-f]\{2\}\).*/\1/' |
      tr -d ':' |
      tr '[:lower:]' '[:upper:]'
}
```

(Note: This code "should" work but has not actually been used to activate a Matlab install before. So it might not give a working host ID after all.)

[matlab_lim]: https://www.mathworks.com/matlabcentral/answers/441674-how-manu-computer-can-use-per-one-account-for-campus-wide-license?s_tid=srchtitle_total%20headcount_8
[vnc]: https://github.com/isi-vista/saga-cluster/wiki/Setting-Up-VNC-Access-for-a-Development-Machine
[lic_cent]: https://www.mathworks.com/licensecenter/licenses/

# Running

## Stroke extraction

To run stroke extraction on the M5 objects with mugs train curriculum, run:

```bash
python adam_preprocessing/shape_stroke_extraction.py \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "path/to/outputs"
```

The outputs will be saved in the usual curriculum format.

Or, using the Slurm script:

```bash
cd adam_preprocessing
sbatch extract_strokes.sh \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "path/to/outputs"
```

## Training

To train the model on say the M5 objects with mugs curriculum, evaluating on the corresponding eval curriculum:

```bash
python adam_preprocessing/shape_stroke_graph_learner.py \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin"
```

Or, using the Slurm script:

```bash
cd adam_preprocessing
sbatch train.sh \
  "data/curriculum/train/m5_objects_v0_with_mugs" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin"
```

Note that neither the Python train script nor the Slurm script handles decode/inference.

## Inference/decode

To run the model trained on M5 objects with mugs curriculum, running decode for the corresponding eval curriculum:

```bash
python adam_preprocessing/shape_stroke_graph_inference.py \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  --save_outputs_to "path/to/outputs"
```

To overwrite the decodes in the input curriculum files:

```bash
python adam_preprocessing/shape_stroke_graph_inference.py \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  --save_outputs_to "data/curriculum/test/m5_objects_v0_with_mugs_eval"
```

To use the Slurm script:

```bash
cd adam_preprocessing
sbatch predict.sh \
  "data/gnn/m5_objects_v0_with_mugs_pytorch.bin" \
  "data/curriculum/test/m5_objects_v0_with_mugs_eval" \
  "path/to/outputs"
```
