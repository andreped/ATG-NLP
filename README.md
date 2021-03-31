# ATG-NLP
Code relevant for training and evaluating NLPs from free text using Auxilliary Task Guiding (ATG)



### Project structure

```
+-- {ATG-NLP}/
|   +-- python/
|   |   +-- create_data.py
|   |   +-- train.py
|   |   +-- [...]
|   +-- data/
|   |   +-- folder_containing_the_dataset/
|   |   |   +-- fold_name0/
|   |   |   +-- fold_name1/
|   |   |   +-- [...]
|   +-- output/
|   |   +-- history/
|   |   |   +--- history_some_run_name1.txt
|   |   |   +--- history_some_run_name2.txt
|   |   |   +--- [...]
|   |   +-- models/
|   |   |   +--- model_some_run_name1.h5
|   |   |   +--- model_some_run_name2.h5
|   |   |   +--- [...]
```


### TODOs (most important from top to bottom):

- [x] Get benchmark datasets using the huggingface/datasets repository
- [x] Setup the project structure
- [x] Make jupyter notebook and code deployable on Google Colab
- [x] Use pretrained BERT as tokenizer
- [x] Introduce simple neural network that performs some task in an end-to-end NLP pipeline
- [ ] Compare BERT with other relevant tokenizers
- [ ] Introduce more benchmark datasets, ideally that are suitable for a specific use case (currently not defined)
- [ ] Implement MTL designs to test hypothesis
- [ ] Write report





------

Made with :heart: and python
