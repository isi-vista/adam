[![Build status](https://travis-ci.com/isi-vista/adam.svg?branch=master)](https://travis-ci.com/isi-vista/adam.svg?branch=master)

[![codecov](https://codecov.io/gh/isi-vista/adam/branch/master/graph/badge.svg)](https://codecov.io/gh/isi-vista/adam)

[![docs](https://readthedocs.org/projects/adam-language-learner/badge/?version=latest)](https://adam-language-learner.readthedocs.io/en/latest/)

# Introduction

ADAM is ISI's effort under DARPA's Grounded Artificial Intelligence Language Acquisition (GAILA) program.
Background for the GAILA program is given in [DARPA's call for proposals](https://www.fbo.gov/utils/view?id=b084633eb2501d60932bb77bf5ffb192)
and [here is a video](https://youtu.be/xGsIKHKqKdw) of a talk giving an overview of our plans for ADAM
(aimed at an audience familiar with the GAILA program).

Documentation can be found [here](https://adam-language-learner.readthedocs.io/en/latest/).
The documentation includes tutorials on how to use ADAM (though it may not appear on Read the Docs).

You can check out our papers to read more about the system and to cite our work:

1. [ADAM: A Sandbox for Implementing Language Learning](https://arxiv.org/abs/2105.02263)
2. [A Grounded Approach to Modeling Generic Knowledge Acquisition](https://arxiv.org/abs/2105.03207)

# Project Setup

1. Create a Python 3.7 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam python=3.7` followed by `conda activate adam`.
2. `pip install -r requirements.txt`
3. Make a file under `parameters` called `root.params` which contains:

   ```
   adam_root: PATH_TO_WORKING_COPY_OF_THIS_REPO
   # if you want to view experiment results in the UI, you must point adam_experiment_root to
   # %adam_root%/data, otherwise any directory is fine:
   adam_experiment_root: %adam_root%/data

   conda_environment: adam
   conda_base_path: PATH_TO_ANACONDA_DIRECTORY
   ```

## Using PyPy

1. Complete the project setup as above.
2. Install PyPy 3.7: `conda install -c conda-forge pypy3.7`.
3. To run tests using PyPy3.7: `make test`.

# Documentation

To generate Sphinx documentation:

```
cd docs
make html
```

The docs will be under `docs/_build/html`

# To generate an HTML dump of the curriculum

Run `adam.curriculum_to_html /full/path/to/parameters/html/curriculum_to_html.phase1.params`

# To Run the Learner over an Entire Curriculum

Run `adam.experiment.run_m9 /full/path/to/parameters/experiment/m9/m9.params`

# To Run the angular viewer application

First, make sure the Flask backend is running. To start the backend:

```bash
cd adam/flask_backend
bash start_flask.sh &
```

The angular application is located under `/angular-viewer/adam-angular-demo`
Navigate to the directory above and serve the application locally on port 4200:

```
ng serve --open
```

# To install npm on your machine

```
npm install -g npm
```

# To install font dependencies on your machine

```
cd {adam_root}
PYTHONPATH=. python scripts/install_fonts.py
```

For more detailed information including how to check npm versions on your machine, please refer: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

# Languages

Currently, our curriculum dump and learner run in English by default, but they are also runnable in Chinese.
Our Chinese implementation uses Yale romanization to maintain UTF-8 encoding, but this can easily be [converted to the more common
Pinyin romanization](https://ctext.org/pinyin.pl?if=en&text=&from=yale&to=pinyin).

To generate a curriculum dump or run the learner in Chinese, add the following line to your `parameters/root.params` file:

```
language_mode : CHINESE
```

and then run the commands given above.

# Contributing

Run `make precommit` before commiting.

If you are using PyCharm, please set your docstring format to "Google" and your unit test runner to "PyTest" in
`Preferences | Tools | Python Integrated Tools`.

# Contributors

- Gayatri Atale
- Deniz Beser
- Joe Cecil
- Marjorie Freedman
- Ryan Gabbard
- Chris Jenkins
- Ashwin Kumar
- Elizabeth Lee
- Jacob Lichtefeld
- Mitch Marcus
- Justin Martin
- Grace McClurg
- Sarah Payne
- Ralph Weischedel
- Charles Yang

# Contact

Deniz Beser (`beser@isi.edu`)
Joe Cecil (`jcecil@isi.edu`)
Ryan Gabbard (`ryan.gabbard@gmail.com`)
Jacob Lichtefeld (`jacobl@isi.edu`)

# Suggested Reading

The following papers have informed our design decisions and may be referenced in issues.

- Biederman, I., 1987. [Recognition-by-components: a theory of human image understanding](https://s3.amazonaws.com/academia.edu.documents/30745513/Recognition_by_Components.pdf?response-content-disposition=inline%3B%20filename%3DRecognition-by-components_a_theory_of_hu.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20191101%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20191101T152508Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=a96e731888ff6e33bce40edf1f7acaf243f3b09556bd72aa77134657913602f1).
  Psychological review, 94(2), p.115-147.
- Dowty, D., 1991. [Thematic Proto-Roles and Argument Selection](http://www.letras.ufmg.br/padrao_cms/documentos/profs/marciacancado/dowty1991.pdf).
  Language, 67(3), pp.547-619
- Jackendoff, R., 1975. [A System of Semantic Primitives](https://www.aclweb.org/anthology/T75-2006.pdf). In
  Theoretical issues in natural language processing.
- Jackendoff, R., 1992. Semantic structures (Vol. 18). MIT press.
- Hart, B. and Risley, T. R., 1995. Meaningful differences in the everyday experience of young American children. Paul
  H Brookes Publishing, Baltimore, MD.
- Landau, B. and Jackendoff, R., 1993. ["What" and "where" in spatial language and spatial cognition](http://www2.denizyuret.com/bib/landau/landau1993and/MLandau.pdf).
  Behavioral and Brain Sciences, 16(2), pp.217-238.
- Marr, D. and Vaina, L., 1982. [Representation and recognition of the movements of shapes](https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.1982.0024).
  Proceedings of the Royal Society of London. Series B. Biological Sciences, 214(1197), pp.501-524.
- Schuler, K.D., Yang, C. and Newport, E.L., 2016. [Testing the Tolerance Principle: Children form productive rules
  when it is more computationally efficient to do so](https://www.ling.upenn.edu/~ycharles/papers/syn2016.pdf). In CogSci.
- Singh, M. and Landau, B., 1998. Parts of visual shape as primitives for categorization. Behavioral and Brain Sciences,
  21(1), pp.36-37.
- Smith, L.B., Jayaraman, S., Clerkin, E. and Yu, C., 2018.
  [The developing infant creates a curriculum for statistical learning](http://www.cogs.indiana.edu/~dll/papers/tics_2018.pdf).
  Trends in cognitive sciences, 22(4), pp.325-336.
- Stevens, J.S., Gleitman, L.R., Trueswell, J.C. and Yang, C., 2017. [The pursuit of word meanings](https://www.ling.upenn.edu/~ycharles/papers/pursuit-final.pdf).
  Cognitive Science, 41, pp.638-676.
- Trueswell, J.C., Lin, Y., Armstrong III, B., Cartmill, E.A., Goldin-Meadow, S. and Gleitman, L.R., 2016.
  [Perceiving referential intent: Dynamics of reference in natural parent–child interactions](https://cpb-us-w2.wpmucdn.com/web.sas.upenn.edu/dist/4/81/files/2017/07/Trueswell-et-al-Perceiving-referential-intent-142dnbw.pdf).
  Cognition, 148, pp.117-135.
- Yang, C., 2005. On productivity. Linguistic variation yearbook, 5(1), pp.265-302.
- Yang, C., 2013. [Ontogeny and Phylogeny of Language](https://www.ling.upenn.edu/~ycharles/PNAS-2013-final.pdf).
  Proceedings of the National Academy of Sciences, 110(16), pp.6324-6327.
