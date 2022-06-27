# Template for extending ASReview with new model

ASReview has support for extensions, which enable you to seemlessly integrate
your own programs with the ASReview framework. These extensions can extend the
software with new classifiers, qurey strategies, balance strategies, and feature
extraction techniques. This template can be used to write such an extension
(add a new SKLearn naive Bayes classifier with default settings in this case).

See the section [Extensions](https://asreview.readthedocs.io/en/latest/API/extensions_dev.html?highlight=extension) 
on ReadTheDocs for more information on writing extensions.

## Getting started

The only package that needs to be installed separately is the following spacy model
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt

```


or

```bash
pip install git@github.com:{USER_NAME}/{REPO_NAME}.git
```

and replace `{USER_NAME}` and `{REPO_NAME}` by your own details. 


## Usage

The new classifier `nb_example` is defined in
[`asreviewcontrib/models/nb_default_param.py`](asreviewcontrib/models/nb_default_param.py) 
and can be used in a simulation.

```bash
asreview simulate example_data_file.csv -m nb_example
```

## License

MIT license
