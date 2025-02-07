# Bird Species Classifier

A deep learning model that can identify 7 bird species using their scientific names:
- Scarlet Macaw (Ara macao)
- Hyacinth Macaw (Anodorhynchus hyacinthinus)
- Blue-and-yellow Macaw (Ara ararauna)
- Toco Toucan (Ramphastos toco)
- Budgerigar (Melopsittacus undulatus)
- Canary (Serinus canaria)
- Cockatiel (Nymphicus hollandicus)

## Live Demo
Try the model here: [Hugging Face Space](https://huggingface.co/spaces/HusseinSerhan/bird-species-classifier)

## About the Project
This classifier was built using:
- fastai and PyTorch for the deep learning model
- Gradio for the web interface
- Scientific names for improved data quality
- Iterative data cleaning using model feedback

## Technical Details
- Model: ResNet18
- Image size: 224x224
- Data augmentation: RandomResizedCrop and aug_transforms
- Deployment: Hugging Face Spaces with Gradio

## Usage
Upload an image of any of these bird species to get:
1. The predicted species
2. Confidence level for the prediction
3. Alternative predictions if the model is uncertain

## Development Process
1. Data collection using scientific names
2. Iterative data cleaning
3. Model training with fastai
4. Deployment with Gradio and Hugging Face Spaces

## Files
- `app.py`: Gradio interface code
- `requirements.txt`: Python dependencies
- `bird_classifier.pkl`: Trained model file
