import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for
import torch
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

app = Flask(__name__)

# Global variables
product_data = None
tokenizer = None
model = None

def load_data_and_model():
    global product_data, tokenizer, model
    
    # Load dataset
    dataset = load_dataset("LoganKells/amazon_product_reviews_video_games")
    product_data = dataset["train"].to_pandas()
    
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def generate_word_cloud(text):
    if not text.strip():
        return None
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    global product_data, tokenizer, model
    
    # Load data and model if not already loaded
    if product_data is None or tokenizer is None or model is None:
        load_data_and_model()
    
    result = None
    error = None
    
    if request.method == 'POST':
        if 'refresh' in request.form:
            return redirect(url_for('index'))
            
        if 'generate' in request.form:
            product_code = request.form.get('product_code', '').strip()
            
            # Validate product code (10 digits)
            if not product_code or not product_code.isdigit() or len(product_code) != 10:
                error = "Please enter a 10 digit number product code."
            else:
                # Get reviews for the product
                product_reviews = product_data[product_data['asin'] == product_code]['reviewText'].tolist()
                
                if not product_reviews:
                    error = "Product code not found."
                else:
                    total_reviews = len(product_reviews)
                    positive_count = 0
                    negative_count = 0
                    positive_reviews_text = ""
                    negative_reviews_text = ""

                    # Perform sentiment analysis on each review
                    for review in product_reviews:
                        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        predicted_class_id = logits.argmax().item()
                        label = model.config.id2label[predicted_class_id]
                        
                        if label == 'POSITIVE':
                            positive_count += 1
                            positive_reviews_text += review + " "
                        else:
                            negative_count += 1
                            negative_reviews_text += review + " "

                    positive_percentage = (positive_count / total_reviews) * 100
                    negative_percentage = (negative_count / total_reviews) * 100

                    # Generate word clouds
                    positive_wordcloud = generate_word_cloud(positive_reviews_text)
                    negative_wordcloud = generate_word_cloud(negative_reviews_text)

                    result = {
                        'product_code': product_code,
                        'total_reviews': total_reviews,
                        'positive_percentage': positive_percentage,
                        'negative_percentage': negative_percentage,
                        'positive_wordcloud': positive_wordcloud,
                        'negative_wordcloud': negative_wordcloud
                    }
    
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)