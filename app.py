import gradio as gr
from fastai.vision.all import *

learn = load_learner('bird_classifier.pkl')

def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

title = "Bird Species Classifier"
description = """This model identifies these bird species:
- Scarlet Macaw (Ara macao)
- Hyacinth Macaw (Anodorhynchus hyacinthinus)
- Blue-and-yellow Macaw (Ara ararauna)
- Toco Toucan (Ramphastos toco)
- Budgerigar (Melopsittacus undulatus)
- Canary (Serinus canaria)
- Cockatiel (Nymphicus hollandicus)

Upload an image of any of these birds to see the model's prediction!"""

article = """<p style='text-align: center'>
Created using fastai and Gradio. 
<a href='https://github.com/YOUR_GITHUB_USERNAME/bird-species-classifier'>GitHub Repository</a>
</p>"""

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
    article=article
)

demo.launch()