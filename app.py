import gradio as gr
from fastai.vision.all import *

learn = load_learner('cat.pkl')
labels = learn.dls.vocab


def classify_image(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


images = gr.inputs.Image(shape=(300, 300))
outputs = gr.outputs.Label(num_top_classes=3)

examples = ['british-shorthair.jpg',
            'maine-coon.jpg', 'european-shorthair.jpg']

interface = gr.Interface(fn=classify_image, inputs=images,
                         outputs=outputs, examples=examples)

interface.launch(inline=False)
