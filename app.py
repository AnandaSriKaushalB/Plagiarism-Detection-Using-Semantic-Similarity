# Import the Flask class from the flask module to create a web application
from flask import Flask, render_template, request
# Import TfidfVectorizer for TF-IDF vectorization and cosine_similarity for calculating similarity scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Import SentenceTransformer for BERT-based semantic similarity and util for cosine similarity calculation
from sentence_transformers import SentenceTransformer, util

# Create a new instance of the Flask class, passing in the current module name
app = Flask(__name__)

# Load a pre-trained BERT model for sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define a function to calculate TF-IDF similarity between two texts
def tfidf_similarity(text1, text2):
    # Initialize a TfidfVectorizer object to convert text data into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer to the text data and transform it into TF-IDF vectors
    tfidf = vectorizer.fit_transform([text1, text2])
    # Calculate the cosine similarity between the TF-IDF vectors of the two texts
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    # Return the similarity score rounded to two decimal places and multiplied by 100
    return round(score * 100, 2)

# Define a function to calculate BERT-based semantic similarity between two texts
def bert_similarity(text1, text2):
    # Encode the input texts into sentence embeddings using the pre-trained BERT model
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    # Calculate the cosine similarity between the sentence embeddings of the two texts
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Return the similarity score rounded to two decimal places and multiplied by 100
    return round(score * 100, 2)

# Define a route for the root URL of the application
@app.route('/')
def home():
    # Render the index.html template when the root URL is accessed
    return render_template('index.html')

# Define a route for the result page that accepts POST requests
@app.route('/result', methods=['POST'])
def result():
    # Get the text inputs from the form data
    text1 = request.form['text1']
    text2 = request.form['text2']

    # Check if both text fields are filled
    if not text1.strip() or not text2.strip():
        # Render the result.html template with an error message if either field is empty
        return render_template('result.html', error="Both text fields are required!")

    # Calculate the TF-IDF similarity score
    tfidf_score = tfidf_similarity(text1, text2)
    # Calculate the BERT-based semantic similarity score
    bert_score = bert_similarity(text1, text2)

    # Render the result.html template with the calculated similarity scores
    return render_template('result.html', tfidf_score=tfidf_score, bert_score=bert_score)

# Check if this script is being run directly (not imported)
if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)