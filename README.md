# PV  Site and Pseudo-Irradience Model Training and Experimentation Code
Template Repository for OCF Projects

## Usage

### Batch Creation

To create new batches, modify the `configs/datapipe/forecastor.yaml` file to point to the appropriate files
Also update `configs/batch.yaml` and `irradiance/batch_writing.py` to write to the correct places and with the correct options.

Then run `python run.py --config-name batch` to create the batches.

To change what data is being created with the datapipe, the changes have to happen to the `ocf_datapipes.training.pseudo_irradiance_datapipe`, which is what handles
the data processing.

### Inference

To run inference, modify `configs/inference_irradiance.yaml` to have the model configuration you want,
and `irradiance/model_inference.py` to run the current-best model on the data that you want.

Then run `python run.py --config-name inference_irradiance` to run inference. By default it uses the 4th GPU on Donatello.

