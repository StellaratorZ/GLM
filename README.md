This is the repository for our  paper"The More You Say, The Less I Know: Large Language Models Suffers From Statistical Hallucinations in Causality Identifica".

## Install Requirements
This repo mainly requires the following packages.
* transformers              4.27.1
* jieba==0.42.1
* cpm_kernels
* torch>=1.10
* gradio
* mdtex2html
* sentencepiece
* accelerate


Full packages are listed in requirements.txt.
```bash
pip3 install -r requirements.txt
```

## Data

The original dataset CMedCausal was obtained from the CHIP release CBLUE, a medical causal entity relationship extraction dataset. It can be accessed through https://tianchi.aliyun.com/dataset/129573.

We processed this dataset using five strategies to obtain datasets that meets the experimental requirements.

### Data Generation Process


We use four approaches to design positive examples with fine causation and negative examples with wrong causation for the event causation identification problems:

- Action1: Swap the head mention and tail mention when they have a relation of causation.
- Action2: Disrupt all mentions for a paragraph, and then split the sentences as in Act1 to get the modified sentences.
- Action3: Selects two causal pairs $A \to B $ and $C \to D $ and swaps the mentions in the two causal pairs once, with the rule that only mentions with the same relative position are swapped. For example, swap BD to form $A\to D$ and $C\to B$, and swap AC to form $A\to D$ and $C\to B$. Then put the new relation into the original sentence.
- Action4: We adopt two strategies to form a control group. The first strategy is to randomly disrupt the mentions in the sentences. we collect all the mentions in the dataset and randomly select mentions to fill in the position where the original sentence mentions are located. The second strategy is to randomly disrupt the words in a sentence. For a sentence, we randomly select a number of words in the sentence and randomly disrupt their positions to get a new sentence.
- Auxiliary_task: Extract the causal pairs that have appeared in the dataset.

## How to Use Our Code

If you have a new dataset with a similar format to  Act_input.json, you can run our code to generate your own causal test dataset.

An example: 

```
python Act1.py
python Act2.py --input_file  Act_input.json
python Act3.py --input_file  Act_input.json
python Act4_shuffle_mention.py --input_file  Act3_5_delet_lessmention_test.json
python Act4_shuffle_word.py --input_file  Act_input.json
python Aux.py
```

### The ask code for GLM and gpt

We provide two official model (`ChatGlm` & `ChatGPT`) for test on dataset,you should change following options with your own path and filename
  ```json
  {
    filename = "YOUR-DATA-FILE"
    base_dir = "YOUR-BASE-DIR"
    result_filename = "YOUR-RESULT-FILE"
  }
  ```

- Test the dataset in ChatGLM model
  ```
  python AskGLM.py
  ```
- Test the dataset in ChatGPT model
  ```
  python AskGPT.py
  ```



## More Questions

If you have more questions, please feel free to submit a GitHub issue.



