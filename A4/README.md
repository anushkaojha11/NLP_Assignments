# NLP Assignment A4 – BERT & Sentence-BERT Implementation

This project implements:

- BERT from scratch (Task 1)
- Sentence-BERT style classification for NLI (Task 2)
- Model comparison with a pre-trained transformer (Task 3)
- A Dash-based web application for real-time inference

---

# Classification Report

The following table summarizes the performance of our Sentence-BERT model on the validation dataset. The evaluation metrics include Precision, Recall, F1-Score, and Support for each class.

| Class                  | Precision | Recall | F1-Score       | Support |
| ---------------------- | --------- | ------ | -------------- | ------- |
| Entailment             | 0.00      | 0.00   | 0.00           | 62      |
| Neutral                | 0.44      | 0.10   | 0.16           | 72      |
| Contradiction          | 0.33      | 0.92   | 0.49           | 66      |
| **Accuracy**     |           |        | **0.34** | 200     |
| **Macro Avg**    | 0.26      | 0.34   | 0.22           | 200     |
| **Weighted Avg** | 0.27      | 0.34   | 0.22           | 200     |

---

## Explanation for Zero Scores in Entailment

The entailment class shows zero precision, recall, and F1-score because the model did not predict any samples as entailment during evaluation. When a class receives no predicted instances, precision becomes undefined and is automatically set to zero by the evaluation metric.

This indicates that the classifier predominantly predicted the contradiction class. The high recall (0.92) for contradiction suggests that the model learned to favor this class, likely because it was easier to separate from the others using the learned sentence embeddings.

Possible reasons for this behavior include:

1. **Limited Training Data** – The model was trained on a relatively small subset of the dataset, reducing its ability to generalize across all classes.
2. **Embedding Quality** – Since the base BERT model was trained from scratch on a limited corpus, the sentence embeddings may not capture nuanced semantic relationships required for entailment detection.
3. **Class Prediction Bias** – During training, the classifier may have converged toward predicting the dominant or easiest class to minimize loss.

Overall, the results indicate that while the model can strongly detect contradictions, it struggles to distinguish entailment and neutral relationships. This suggests that improved pretraining or larger training data would likely enhance performance.

---

# Comparison of Our Model with Pre-trained Model

To evaluate semantic understanding, we compare our trained model against a pre-trained Sentence Transformer model using cosine similarity.

| Model Type  | Cosine Similarity (Similar sentence) | Cosine Similarity (Dissimilar sentence) |
| ----------- | ------------------------------------ | --------------------------------------- |
| Our Model   | 0.993                                | 0.993                                   |
| Pre-trained | 0.731                                | 0.483                                   |

---

## Interpretation

Both similar and dissimilar sentence pairs resulted in a cosine similarity score of **0.993** for our model. This indicates that the model is not effectively distinguishing between semantically related and unrelated sentence pairs.

Ideally, similar sentences should produce a high cosine similarity score, while dissimilar or contradictory sentences should yield a significantly lower score. Since both scores are nearly identical, this suggests that the sentence embeddings are not sufficiently discriminative.

Possible reasons include:

1. The base BERT model was trained from scratch on a limited dataset, leading to weak semantic representations.
2. The classifier may have overfitted to a dominant class during training.
3. The sentence embedding space may have collapsed, producing highly similar vectors regardless of input.

The pre-trained model, in contrast, demonstrates more reasonable separation between similar and dissimilar sentences, highlighting the importance of large-scale pretraining.

---

# Discussion

The implementation of BERT from scratch was carried out by taking reference from the professor’s provided materials. For pretraining, the Wikipedia dataset from Hugging Face was initially selected. However, due to hardware limitations, it was not feasible to train on the full dataset. As a result, the dataset was filtered down to 100,000 samples for training. After preprocessing and vocabulary construction, the BERT model class was implemented and training was initiated.

During training, significant computational challenges were encountered. Memory constraints required reducing the batch size to 3 and limiting the number of epochs. Initially, the model was tested with 1000 epochs, but the loss showed minimal improvement and eventually the system ran out of memory. Consequently, the training configuration was adjusted to 700 epochs with reduced batch size. As expected, the limited dataset size and constrained training setup negatively affected model performance during inference.

In Task 2, the SNLI and MNLI datasets were used to train a custom Sentence-BERT style model for Natural Language Inference. After preprocessing and tokenization, the model was trained for 5 epochs. Due to hardware limitations, the intended batch size of 32 was reduced to 8. While training completed successfully, the model struggled to generalize well across all classes.

During evaluation and analysis (Task 3), the model’s performance was compared with a pre-trained model from Hugging Face. The comparison clearly demonstrated the performance gap between a model trained from scratch on limited data and a large-scale pre-trained transformer.

The main challenges encountered were:

- Limited model performance due to training from scratch
- Reduced dataset size caused by hardware constraints
- Memory limitations affecting batch size and training stability
- Computational resource restrictions preventing large-scale experimentation

---

## Proposed Improvements

To improve the model’s performance in future work:

- Increase the size of the training dataset
- Utilize more powerful hardware (GPU with larger memory)
- Experiment with larger batch sizes and optimized learning rates
- Increase model depth and hidden dimensions
- Apply transfer learning using a pre-trained base model

---

# Web Application Interface Documentation

For this assignment, the web interface was developed using Dash. The entire user interface and model integration logic are implemented in the `app.py` file. The interface consists of two text input fields, a Predict button, input validation, and a result display section.

The trained model is loaded from `sen_bert_full.pth`, and the saved weights are restored into the BERT model and classifier head. The `BertTokenizer` from `bert-base-uncased` is used for tokenization. The input sentences are converted into embeddings using the `get_last_hidden_state()` method followed by mean pooling. Cosine similarity between the sentence embeddings is computed, and the concatenated vector `[u, v, |u - v|]` is passed to the classifier to predict one of three labels: Entailment, Neutral, or Contradiction.

The user interaction flow is:

- User enters two sentences
- Sentences may express similar, neutral, or opposite meanings
- User clicks the Predict button
- The prediction and cosine similarity score are displayed

## Demo

![Application Demo](demo.gif)
