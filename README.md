# Probing Causality Manipulation of Large Language Models ![rocket](https://github.githubassets.com/images/icons/emoji/unicode/1f680.png?v8)

This is the repository for our paper"Probing Causality Manipulation of Large Language Models".

## ![white_check_mark](https://github.githubassets.com/images/icons/emoji/unicode/2705.png?v8) Abstract

 Large language models (LLMs) have shown various ability on natural language processing, including problems about causality. It is not intuitive for LLMs to command causality, since pretrained models usually work on statistical associations, and do not focus on causes and effects in sentences. So that probing internal manipulation of causality is necessary for LLMs. Our work proposes a novel approach to probe causality manipulation hierarchically, by providing different shortcuts to models and observe behaviors. We exploit retrieval augmented generation (RAG) and in-context learning (ICL) to provide shortcuts. We conduct experiments on mainstream LLMs, including GPT-4 and some smaller and domain-specific models. Our results suggest that LLMs can detect entities related to causality and recognize direct causal relationships. However, LLMs lack specialized cognition for causality, merely treating them as part of the global semantic of the sentence. This restricts for further recognition of causality for LLMs.

## ![ledger](https://github.githubassets.com/images/icons/emoji/unicode/1f4d2.png?v8)Install Requirements

This repo mainly requires the following packages.

- transformers==4.27.1
- jieba==0.42.1
- cpm_kernels
- torch>=1.10
- gradio
- mdtex2html
- sentencepiece
- accelerate

Full packages are listed in requirements.txt.

```bash
pip3 install -r requirements.txt
```

## ![fuelpump](https://github.githubassets.com/images/icons/emoji/unicode/26fd.png?v8) Data

The original dataset CMedCausal was obtained from the CHIP release CBLUE, a medical causal entity relationship extraction dataset. It can be accessed through https://tianchi.aliyun.com/dataset/129573.

We processed this dataset using five strategies to obtain datasets that meets the experimental requirements.

### Data Generation Process


We use four approaches to design positive examples with fine causation and negative examples with wrong causation for the event causation identification problems:

- Action1: Swap the head mention and tail mention when they have a relation of causation.
- Action2: Disrupt all mentions for a paragraph, and then split the sentences as in Act1 to get the modified sentences.
- Action3: Selects two causal pairs $A \to B $ and $C \to D $ and swaps the mentions in the two causal pairs once, with the rule that only mentions with the same relative position are swapped. For example, swap BD to form $A\to D$ and $C\to B$, and swap AC to form $A\to D$ and $C\to B$. Then put the new relation into the original sentence.
- Action4: We adopt two strategies to form a control group. The first strategy is to randomly disrupt the mentions in the sentences. we collect all the mentions in the dataset and randomly select mentions to fill in the position where the original sentence mentions are located. The second strategy is to randomly disrupt the words in a sentence. For a sentence, we randomly select a number of words in the sentence and randomly disrupt their positions to get a new sentence.
- Auxiliary_task: Extract the causal pairs that have appeared in the dataset.

## ![factory](https://github.githubassets.com/images/icons/emoji/unicode/1f3ed.png?v8) How to Use Our Code

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



## ![crystal_ball](https://github.githubassets.com/images/icons/emoji/unicode/1f52e.png?v8)More Questions

If you have more questions, please feel free to submit a GitHub issue.



