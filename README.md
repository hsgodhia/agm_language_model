## About
An implementation of Aggregate and mixed-order Markov models for statistical language processing as described in the Saul & Pereira 1997 paper.

Author
------
Harshal Godhia  
hgodhia@umass.edu

Files
-----
`lm.py`  
Contains all the python code for training and testing the model.

`model_params.pkl`  
A pre-trained model whose parameters are serialized and saved into this file

`test_sentences.txt`  
Sample sentences used througout the questions for the homework

FAQ
--------

*Q1* Which file do I run?  
*A1* `lm.py` will use a pre-trained model and sample sentences from the file test_sentences.txt to evaluate sentence probability

*Q2* What packages do i need to run?  
*A2*  This code is tested in the environment Python 3.6.0 64-bit. You would need the following packages installed nltk, nltk.data(brown corpus), numpy, matplotlib


*Q3*Where do I add new test sentence whose probability I want?  
*A3*Add new sentences, one sentence per line, in the file test_sentences.txt. the code will report sentence probability for the sentence and its reverse ordering. Incase you want a custom ordering then use add the delimiter || on the same line and another sentence.
- Some details
 - There are lots of sentences already prepopulated (that can serve as examples if you like to add more test cases)
 - Please ensure all words are in the vocabulary, it will throw an exception otherwise (not handled by code)

*Q4* What is model_params.pkl  
*A4* It is a pre-trained model (emission and transition matrices) which I found to give good numbers for the questions in the assignment. Since EM approaches a local optimum and depends on random initialization, the results like the ratio for the colorless sentence may not be reproduced. Hence I saved the best model (after different tries) in this file.

*Q5* How do I train a new model?  
*A5* Uncomment line #353. It will train a new model and save it into the file model_params.pkl replacing the pre-trained model

- Some details
 - Training may take upto 4 mins overall
 - Please re-run incase the results don't match as reported in the writeup