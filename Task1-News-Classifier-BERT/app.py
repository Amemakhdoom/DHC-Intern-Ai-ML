import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./bert-news-classifier",
    tokenizer="./bert-news-classifier",
    device=-1
)

label_map = {
    "LABEL_0": "🌍 World",
    "LABEL_1": "⚽ Sports",
    "LABEL_2": "💼 Business",
    "LABEL_3": "💻 Sci/Tech"
}

def predict(headline):
    result = classifier(headline)[0]
    category = label_map[result["label"]]
    confidence = round(result["score"] * 100, 2)
    return f"{category}  —  Confidence: {confidence}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Enter a news headline here...",
        label="News Headline"
    ),
    outputs=gr.Textbox(label="Predicted Category"),
    title="📰 News Topic Classifier",
    description="Powered by Fine-Tuned BERT on TPU",
    examples=[
        ["Federer wins Wimbledon championship for the 7th time"],
        ["Apple reports record breaking quarterly revenue"],
        ["NASA launches new Mars rover to explore red planet"],
        ["United Nations holds emergency meeting over conflict"],
    ]
)

demo.launch(share=True)
