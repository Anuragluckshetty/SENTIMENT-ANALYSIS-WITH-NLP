# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIIONS

*NAME*: ANURAG LUCKSHETTY

*INTERN ID*: CTIS1612

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

##In this task, sentiment analysis was performed on a dataset of customer reviews using Natural Language Processing (NLP) techniques. Sentiment analysis is a text classification problem that aims to determine the emotional tone or opinion expressed in a piece of text. In this case, the objective was to classify customer reviews into two categories: positive sentiment and negative sentiment. This task demonstrates how textual data can be processed and analyzed using machine learning models.

The first step in the task involved preparing the dataset. A set of customer reviews was used, where each review was associated with a sentiment label. Positive reviews were labeled as 1, while negative reviews were labeled as 0. Since machine learning algorithms cannot directly work with raw text, preprocessing was an essential step. The text preprocessing phase included converting all text to lowercase to maintain consistency, removing punctuation marks and special characters, and eliminating unnecessary symbols. These steps help reduce noise in the data and improve the quality of features extracted from the text.

After preprocessing, the cleaned textual data needed to be transformed into numerical form. For this purpose, the TF-IDF (Term Frequency–Inverse Document Frequency) vectorization technique was used. TF-IDF converts text into numerical vectors by measuring how important a word is to a document relative to the entire dataset. Words that appear frequently in a specific review but are rare across all reviews receive higher weights, while very common words receive lower importance. This technique helps capture meaningful patterns in text while reducing the influence of common but less informative words.

Once the text data was vectorized, the dataset was split into training and testing sets. Eighty percent of the data was used for training the model, and twenty percent was used for testing. This split ensures that the model’s performance is evaluated on unseen data, which helps assess its ability to generalize rather than memorize the training samples.

A Logistic Regression classifier was then used for sentiment classification. Logistic Regression is a widely used algorithm for binary classification problems and works well with high-dimensional sparse data such as TF-IDF vectors. The model was trained on the TF-IDF features of the training dataset and learned patterns that distinguish positive reviews from negative ones. After training, the model was used to predict sentiments for the test dataset.

The performance of the model was evaluated using accuracy and a classification report. Accuracy measures the proportion of correctly classified reviews, while the classification report provides detailed metrics such as precision, recall, and F1-score for each class. These evaluation metrics indicate how effectively the model identifies positive and negative sentiments.

In conclusion, this task successfully demonstrates the complete pipeline of sentiment analysis using NLP techniques. By combining text preprocessing, TF-IDF vectorization, and Logistic Regression, the model is able to classify customer reviews with good performance. This task highlights the effectiveness of traditional machine learning methods for text classification problems and provides a strong foundation for understanding more advanced NLP models.
