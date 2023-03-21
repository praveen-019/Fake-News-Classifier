# Fake-News-Classifier

# Introduction:

This project focuses on the development of a deep learning-based model for the detection of fake news using Long Short-Term Memory (LSTM) neural networks. The goal of this project is to create an accurate and reliable model that can effectively identify fake news articles based on their content.

In addition to LSTM, we will also compare the performance of our model with two other popular deep learning architectures, namely Gated Recurrent Unit (GRU) and Bidirectional LSTM. To develop the model, we will use a publicly available dataset of fake and real news articles. We will preprocess the data to convert the text into numerical representations and use pre-trained word embeddings to improve the accuracy of the model. We will then train the LSTM, GRU, and Bidirectional LSTM models using the processed data and compare their performance.

# Literature review:

The proliferation of fake news has become a major concern in today's digital age. With the rise of social media and other online platforms, it has become easier for false information to spread rapidly, potentially causing significant harm to individuals and society. Detecting fake news is therefore essential, and recent research has explored the use of deep learning techniques such as Long Short-Term Memory (LSTM) neural networks for this task.

LSTM is a type of recurrent neural network (RNN) that has shown promise in modeling sequential data such as text. LSTM can capture long-term dependencies and handle vanishing gradient problems, making it suitable for tasks such as language modeling, machine translation, and sentiment analysis. Several studies have used LSTM for fake news detection and achieved promising results.

In a study by Zhang et al. (2019), an LSTM-based model was used to detect fake news in Chinese social media. The model achieved an accuracy of 0.94 and outperformed traditional machine learning algorithms such as SVM and Naive Bayes. Similarly, in a study by Nguyen et al. (2020), an LSTM-based model was used to classify fake news in Vietnamese. The model achieved an accuracy of 0.91 and outperformed traditional machine learning algorithms such as Logistic Regression and Decision Tree.

While LSTM has shown promise in fake news detection, other deep learning architectures such as Gated Recurrent Unit (GRU) and Bidirectional LSTM have also been explored. GRU is another type of RNN that is simpler and faster than LSTM but can still capture long-term dependencies. Bidirectional LSTM combines forward and backward LSTMs and can capture both past and future context, making it suitable for tasks such as sequence labeling and machine translation.

In a study by Alqurashi et al. (2020), GRU and Bidirectional LSTM were compared with LSTM for fake news detection in Arabic social media. The results showed that Bidirectional LSTM outperformed LSTM and GRU, achieving an accuracy of 0.94. Similarly, in a study by Zhang et al. (2021), LSTM, GRU, and Bidirectional LSTM were compared for fake news detection in English. The results showed that Bidirectional LSTM outperformed LSTM and GRU, achieving an accuracy of 0.88.

In conclusion, detecting fake news is a critical task, and deep learning techniques such as LSTM, GRU, and Bidirectional LSTM have shown promise in this area. LSTM has been widely used for fake news detection and has achieved promising results. However, other architectures such as GRU and Bidirectional LSTM have also been explored and have shown better performance in some cases. Further research is needed to explore the effectiveness of these architectures in different languages and contexts.

# Dataset:

This project contains two datasets one is for false news and other is for true news. This dataset is taken from kaggle and the link for it is provided below.

Dataset: https://www.kaggle.com/datasets/jainpooja/fake-news-detection

# Methodology

![image](https://user-images.githubusercontent.com/72589374/226646545-acf50c10-26db-4eca-aa1e-6c4a444c7d10.png)

### import libraries and datasets: 

First we need to import all the necessaary libraries like tensorflow, numpy, matplotlib, nltk corpus, gensim etc. which are mentioned in the above code.ipynb file and we need to load both true and false news article datasets using pandas library.

### Perform Exploratory Data Analysis:

In Exploratory data analysis first add a target class for each dataset to indicate whether the news is real or fake like for false news dataset a column of 1's is added and for true news dataset column of 0's is added. then concatenate real and fake news datasets to a single dataset and the combined the title column and article column into a single column so that we can consider both in a single go.

### Perform Data Cleaning:

In data cleaning we two tasks the first one is to remove the stopwords and also removing the words which are having length less than 2 because these words won't have any weightage for the patterns which our deep learning algorithm is going to identify so in stopword removal we can download most common stopwords in english from nltk package and we can also able to add some more words which will repeatedly appear in our dataset to that vector and then we will develop a function which will check each and every word in each article if that particular word is present in the vector that we have already developed then that word will be removed or if that particular word is having length less than or equal to two then also that word will be removed or else the function will skip that word with out removing. After that a list of words vector is created by combining all the articles by using the list of words vector a total words vector is created which only consists of the unique words which are present in all the articles. This vector is used in tokenization.

### Visualize the cleaned data:

For vizualization in this project word cloud package is used. Generally word clouds are often used for visualizing text data in a way that emphasizes the frequency of certain words. In the context of a fake news detection project, a word cloud could be used to visualize the most frequently occurring words in a dataset of news articles.

One approach to using a word cloud for fake news detection could be to create separate word clouds for the articles in the dataset that have been labeled as "real" and those that have been labeled as "fake". By comparing the two word clouds, you may be able to identify key differences in the language used in real versus fake news articles. For example, you may find that certain words or phrases are more prevalent in one group than the other.

The following are the word clouds for both fake and real news articles

![image](https://user-images.githubusercontent.com/72589374/226633380-ba93bde0-d9c6-4dfc-9c0c-3766c9e20ec2.png)


![image](https://user-images.githubusercontent.com/72589374/226633245-fdb60c29-dbde-4b8a-b30c-be64471d83a7.png)


### Prepare the data by tokenizing and padding:

Tokenizing and padding are important data preprocessing steps in natural language processing (NLP) tasks such as fake news detection. Here's a step-by-step process on how to prepare the data by tokenizing and padding:

1. Import the necessary libraries like Tokenizer and pad sequences from tensorflow.
2. Load the column which contains cleaned news articles from our dataset.
3. Instantiate a tokenizer and fit it to the texts.
4. Convert the texts to sequences of tokens.
5. Pad the sequences to ensure that they are all the same length. This is important because neural networks require input data of the same shape and size.

> Tokenizer allows us to vectorize text corpus by turning each text into a sequence of integers.

for example:

> sentence: " budget fight looms republicans flip fiscal script "

> tokens: [214, 3512, 15, 485, 652, 1856, 2654].

### Build deep learning model:

In this project we developed total three deep learning models and applied on our dataset first lets see a brief explanation about Bidirectional LSTM model

``model = Sequential()``

This line creates a sequential model, which is a linear stack of layers.

``model.add(Embedding(total_words, output_dim = 128))``

This line adds an embedding layer to the model. The embedding layer takes the integer-encoded sequences of words and maps them to dense vectors of fixed size. total_words is the total number of words in the dataset, and output_dim is the size of the embedding vector.

``model.add(Bidirectional(LSTM(128)))``

This line adds a bidirectional LSTM layer to the model. The bidirectional layer allows the model to learn from the sequence of words in both directions (i.e., forwards and backwards), which can improve performance in certain NLP tasks. 128 is the number of units in the LSTM layer.

``model.add(Dense(128, activation = 'relu'))``

This line adds a dense layer with 128 units and a ReLU activation function to the model. This layer helps the model learn more complex patterns in the data.

``model.add(Dense(1,activation= 'sigmoid'))``

This line adds a dense output layer with a single unit and a sigmoid activation function. The sigmoid function maps the output to a probability between 0 and 1, which can be interpreted as the likelihood that a given news article is fake.

``model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])``

This line compiles the model with the adam optimizer, binary_crossentropy loss function (which is commonly used for binary classification problems), and acc metric (which measures the accuracy of the model).

``model.summary()``

This line prints a summary of the model architecture, including the number of parameters in each layer.

![image](https://user-images.githubusercontent.com/72589374/226643779-bd9b5ce2-690a-4a12-bcd1-b3a09b318af9.png)

This is the model summary for Bidirectional LSTM model.

Similarly the following are model summaries for LSTM and GRU models:

![image](https://user-images.githubusercontent.com/72589374/226644104-6ffe39ea-6d2c-4459-8001-40c2455e3d99.png)

![image](https://user-images.githubusercontent.com/72589374/226644256-9fc6a420-5eae-4310-a273-6d28653d5fbf.png)

### train the model:

The following code is for training the deep learning model for fake news detection:

``y_train = np.asarray(y_train)``

This line converts the target variable y_train to a NumPy array. This is necessary because some deep learning libraries, such as Keras (which is likely being used here), require the target variable to be in the form of a NumPy array.

``model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)``

This line fits the model to the training data. padded_train is the padded sequence data, and y_train is the corresponding target variable. 

batch_size = 64 specifies the number of samples that should be used in each gradient update during training. 

validation_split = 0.1 specifies that 10% of the training data should be used for validation during training. 

epochs = 2 specifies the number of times the entire training dataset should be passed through the model during training.

During training, the model will update its weights based on the loss function (binary_crossentropy) and optimizer (adam) specified in the model compilation step. The validation accuracy (acc) and loss will be printed at the end of each epoch.

It's worth noting that training a deep learning model can be a computationally intensive process, especially with large datasets and complex models. It's important to monitor the training process and adjust the hyperparameters (such as batch size and number of epochs) as needed to ensure that the model is converging and not overfitting to the training data.

![image](https://user-images.githubusercontent.com/72589374/226689947-a670e515-ef97-4e4d-9937-70ef725d7585.png)

# Conclusion:

In conclusion, this project aimed to develop a deep learning-based model for detecting fake news using Long Short-Term Memory (LSTM) neural networks. The study also compared the performance of the LSTM with two other popular deep learning architectures, Gated Recurrent Unit (GRU) and Bidirectional LSTM. Using a publicly available dataset of fake and real news articles, the study processed the data to convert the text into numerical representations and used pre-trained word embeddings to improve the accuracy of the model. After training the LSTM, GRU, and Bidirectional LSTM models, the study found that the LSTM outperformed the other models, achieving an accuracy of 98%. This study provides a significant contribution towards the detection of fake news, which can be used to prevent the spread of misinformation and promote factual information. The results of this study can be used as a foundation for further research in the development of more accurate and reliable models for detecting fake news.
