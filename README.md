# Machine Translation
Machine translation project for the SPP 'Machine Translation' by RWTH computer science department i6 in the summer semester 2023. <br>
## Creators: <br>
<ul>
    <li>Andreas Pletschko : andreas.pletschko@rwth-aachen.de
    <li>Lukas Vierling : lukas.vierling@rwht-aachen.de
    <li>Glen Grant : glen.grant@rwth-aachen.de
    <li>Justus Peretti : justus.peretti@rwth-aachen.de
</ul>

## Supervisors: <br>
<ul>
    <li>Benedikt Hilmes
    <li>Prof. Dr.-Ing. Hermann Ney
</ul>

## Usage
To train the model, execute the following command:
```
python main.py train 
```
You can either pass hyperparameters as arguments with the python script, e.g. :
```
python main.py train --epoch 10 --optimizer adam
```
Alternatively, you can provide a predefined YAML config file:
```
python main.py train --config <path to config file>
```
To train an existing model, use the following command:
```
python main.py train --model <path to model>
```

## Part 1
Scoring methods are important for machine translation because they provide a way to measure the accuracy and quality of the translation output. This helps to identify areas of improvement and evaluate the performance of different translation models. Additionally, scoring methods are necessary to compare the translation output to the reference or human-generated translations, which is essential for benchmarking and evaluation of machine translation systems.<br>
Implementation of several scoring methods to compare hypothesis and references. <br>
<ul>
    <li>WER (<b>W</b>ord <b>E</b>rror <b>R</b>ate)
    <li>PER (<b>P</b>osition-independent <b>E</b>rror <b>R</b>ate)
    <li>BLEU (<b>B</b>i<b>l</b>ingual <b>E</b>valuation <b>U</b>nderstudy)
    <li>Levenshtein-Distance
</ul>

## Part 2
Byte Pair Encoding (BPE) is a tokenization algorithm that splits words into subwords based on their frequency in a given text corpus. BPE is an important preprocessing step for many NLP tasks, as it can reduce the vocabulary size and improve model performance. In addition, batching is a crucial technique for efficient training of neural networks, as it allows for parallel processing of multiple input samples. Together, BPE and batching can significantly improve the speed and accuracy of NLP models, making them more practical and scalable for real-world applications. For example, BERT, one of the most successful NLP models, utilizes BPE and batching to achieve state-of-the-art results on a variety of NLP benchmarks, demonstrating the importance of these techniques for the advancement of natural language understanding.<br>
Implementation of several preprocessing steps.<br>
<ul>
    <li><b>B</b>yte <b>P</b>air <b>E</b>ncoding (BPE)
    <li>A Dictionary
    <li>Batch Function
</ul>

## Part 3
A first simple neural model for translating from German to English sentences is implemented. Model is implemented using torch, we then write a training script to learn the model's weights.
Finally, we tune the model's hyperparameters and experiment with different architectures to achieve the best possible perplexity on dev set.
<ul>
    <li>Training on batches created in Part 2
    <li>Saving and loading models
    <li>Evaluating model on development data periodically
    <li>Printing architecture of the model
    <li>Learning rate scheduling
    <li>Hyper parameter tuning
</ul>

![Model Architecture](results/FFModel.png)


