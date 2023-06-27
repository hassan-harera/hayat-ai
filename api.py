from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the classifier model from the pickle file
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request body
    text = data['text']  # Extract the 'text' field from the JSON data
    text = Tfidf(text)

    # Make predictions using the classifier
    label = int(classifier.predict(text)[0])
    certainty = (np.array(classifier.predict_proba(text)[0])[label])

    if label == 0:
        label = 'BOOK'
    elif label == 1:
        label = 'CLOTHING'
    elif label == 2:
        label = 'FOOD'
    elif label == 3:
        label = 'MEDICINE'
    elif label == 4:
        label = 'POSSESSIONS'
    else:
        label = 'other'

    response = {
        'label': label,
        'certainty': float(certainty)
    }

    return jsonify(response)


idf_values = np.load('idf_values.npy')

with open('vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)


def Tfidf(new_text):
    # Use the IDF values to weight the importance of each token.
    new_text_vector = [0] * len(vocabulary)
    for word in new_text.split():
        if word in vocabulary:
            new_text_vector[vocabulary[word]] += 1

    weighted_tokens = np.multiply(new_text_vector, idf_values).reshape(1, len(vocabulary))

    return weighted_tokens


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
