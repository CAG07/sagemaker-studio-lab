# Getting Started with Amazon SageMaker Studio Lab
## Welcome to your SageMaker Studio Lab project
Your SageMaker Studio Lab project is a notebook development environment with 15 GB of persistent storage and access to a CPU or GPU runtime. Everything about your project is automatically saved (notebooks, source code files, datasets, Git repos you have cloned, Conda environments, JupyterLab extensions, etc.) so each time you launch your project you can pick up where you left off. SageMaker Studio Lab is based on the open-source JupyterLab, so you can take advantage of open-source Jupyter extensions in your project.
## Running Python code
This Getting Started document is a [Jupyter notebook](https://jupyter.org/). Notebooks enable you to combine live code, equations, Markdown, images, visualizations, and other content into a single document that you can share with other people. 

To run the following Python code, select the cell with a click, and then type `Shift-Enter` on your keyboard or click the play button in the notebook toolbar at the top of the document.
a = 10
b = 20
c = a + b
print(a, b, c)
To learn more about Python see [The Python Tutorial](https://docs.python.org/3/tutorial/).
## SageMaker Distribution Images
Amazon SageMaker Studio Lab now supports **SageMaker Distribution images**, which provide a richer set of libraries that aren't present in the default environment. These images also provide consistency with environments available in Amazon SageMaker Studio. 

The inclusion of SageMaker distribution images does not impact any existing persistent environments. Notebooks that are using the default environment will continue to work as before.

To create a notebook using these images, see "Creating notebooks, source code files, and accessing the Terminal" section below
## Creating notebooks, source code files, and accessing the Terminal
SageMaker Studio Lab lets you create notebooks, source code files, and access the built-in Terminal. You can do this by clicking on the "+" button at the top of the file browser in the left panel to open the Launcher:

![Launcher Button](images/launcher_button.png)
In the Launcher, there are a set of cards that allow you to launch notebooks in different environments, create source code files, or access the Terminal:

![Launcher Cards](images/launcher_cards_wsmdistro.png)
We recommend creating notebooks using the **sagemaker-distribution** environment, which includes a rich set of pre-installed libraries commonly used for data engineering and AI/ML tasks.

All of the notebooks, files, and datasets that you create are saved in your persistent project directory and are available when you open your project. To get help or access documentation, click on the **Help** menu in the menu bar at the top of the page.
## Installing Python packages
The simplest way of installing Python packages is to use either of the following magic commands in a code cell of a notebook:

`%conda install <package>`

`%pip install <package>`

These magic commands will always install packages into the environment used by that notebook and any packages you install are saved in your persistent project directory. Note: we don't recommend using `!pip` or `!conda` as those can behave in unexpected ways when you have multiple environments.

Here is an example that shows how to install NumPy into the environment used by this notebook:
%conda install numpy
Now you can use NumPy:
import numpy as np
np.random.rand(10)
## SageMaker Studio Lab example notebooks
SageMaker Studio Lab works with familiar open-source data science and machine learning libraries, such as [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/). 

To help you take the next steps, we have a GitHub repository with a set of example notebooks that cover a wide range of data science and machine learning topics, from importing and cleaning data to data visualization and training machine learning models.

<button class="jp-mod-styled" data-commandlinker-command="git:clone" data-commandlinker-args="{&quot;URL&quot;: &quot;https://github.com/aws/studio-lab-examples.git&quot;}">Clone SageMaker Studio Lab Example Notebooks</button>
## AWS Machine Learning University

[Machine Learning University (MLU)](https://aws.amazon.com/machine-learning/mlu/) provides anybody, anywhere, at any time access to the same machine learning courses used to train Amazon’s own developers on machine learning. Learn how to use ML with the learn-at-your-own-pace MLU Accelerator learning series.

<button class="jp-mod-styled" data-commandlinker-command="git:clone" data-commandlinker-args="{&quot;URL&quot;: &quot;https://github.com/aws-samples/aws-machine-learning-university-accelerated-tab.git&quot;}">Clone MLU Notebooks</button>
## Dive into Deep Learning (D2L)

[Dive into Deep Learning (D2L)](https://www.d2l.ai/) is an open-source, interactive book that teaches the ideas, the mathematical theory, and the code that powers deep learning. With over 150 Jupyter notebooks, D2L provides a comprehensive overview of deep learning principles and a state-of-the-art introduction to deep learning in computer vision and natural language processing. With tens of millions of online page views, D2L has been adopted for teaching by over 300 universities from 55 countries, including Stanford, MIT, Harvard, and Cambridge.
    
<button class="jp-mod-styled" data-commandlinker-command="git:clone" data-commandlinker-args="{&quot;URL&quot;: &quot;https://github.com/d2l-ai/d2l-pytorch-sagemaker-studio-lab.git&quot;}">Clone D2L Notebooks</button>
## Hugging Face

[Hugging Face](http://huggingface.co/) is the home of the [Transformers](https://huggingface.co/transformers/) library and state-of-the-art natural language processing, speech, and computer vision models.

<button class="jp-mod-styled" data-commandlinker-command="git:clone" data-commandlinker-args="{&quot;URL&quot;: &quot;https://github.com/huggingface/notebooks.git&quot;}">Clone Hugging Face Notebooks</button>
## Switching to a GPU runtime
Depending on the kinds of algorithms you are using, you may want to switch to a GPU or a CPU runtime for faster computation. First, save your work and then navigate back to your project overview page to select the instance type you want. You can navigate back to your project page by selecting the **Open Project Overview Page** in the **Amazon SageMaker Studio Lab** menu. Switching the runtime will stop all your kernels, but all of your notebooks, files, and datasets will be saved in your persistent project directory.

Note that a GPU runtime session is limited to 4 hours and a CPU runtime session is limited to 12 hours of continuous use.
## Managing packages and Conda environments
### Your default environment

SageMaker Studio Lab uses Conda environments to encapsulate the software (Python, R, etc.) packages needed to run notebooks. Your project contains a default Conda environment, named `default`, with the [IPython kernel](https://ipython.readthedocs.io/en/stable/) and that is about it. There are a couple of ways to install additional packages into this environment.

As described above, you can use the following magic commands in any notebook:

`%conda install <package>`

`%pip install <package>`

These magic commands will always install packages into the environment used by that notebook and any packages you install are saved in your persistent project directory. Note: we don't recommend using `!pip` or `!conda` as those can behave in unexpected ways when you have multiple environments.

Alternatively, you can open the Terminal and activate the environment using:

`$ conda activate default`

Once the environment is activated, you can install packages using the [Conda](https://docs.conda.io/en/latest/) or [pip](https://pip.pypa.io/en/stable/) command lines:

`$ conda install <package>`

`$ pip install <package>`

The conda installation for SageMaker Studio Lab uses a default channel of [conda-forge](https://conda-forge.org/), so you don't need to add the `-c conda-forge` argument when calling `conda install`.
### Creating and using new Conda environments

There are a couple of ways of creating new Conda environments.

**First**, you can open the Terminal and directly create a new environment using the Conda command line:

`$ conda env create --name my_environment python=3.9`

This example creates an new environment named `my_environment` with Python 3.9.

**Alternatively**, if you have a Conda environment file, can right click on the file in the JupyterLab file browser, and select the "Build Conda Environment" item:

![Create Environment](images/create_environment.png)

To activate any Conda environment in the Terminal, run:

`$ conda activate my_environment`

Once you do this, any pakcages installed using Conda or pip will be installed in that environment.

To use your new Conda environments with notebooks, make sure the `ipykernel` package is installed into that environment:

`$ conda install ipykernel`

Once installed `ipykernel`, you should see a card in the launcher for that environment and kernel after about a minute.

<div class="alert alert-info"> <b>Note:</b> It may take about one minute for the new environment to appear as a kernel option.</div>
## Installing JupyterLab and Jupyter Server extensions
SageMaker Studio Lab enables you to install open-source JupyterLab and Jupyter Server extensions. These extensions are typically Python packages that can be installed using `conda` or `pip`. To install these extensions, open the Terminal and activate the `studiolab` environment:

`$ conda activate studiolab`

Then you can install the relevant JupyterLab or Jupyter Server extension:

`$ conda install <jupyter_extension>`

You will need to refresh your page to pickup any JupyterLab extensions you have installed, or power cycle your project runtime to pickup any Jupyter server extensions. 
## Adding *Open in Studio Lab* links to your GitHub repositories
If you have public GitHub repositories with Jupyter Notebooks, you can make it easy for other users to open these notebooks in SageMaker Studio Lab by adding an *Open in Studio Lab* link to a README.md or notebook. This allows anyone to quickly preview the notebook and import it into their SageMaker Studio Lab project.

To add an *Open in Studio Lab* badge to your README.md file use the following markdown

```
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/org/repo/blob/master/path/to/notebook.ipynb)
```

and replace `org`, `repo`, the path and the notebook filename with those for your repo. Or in HTML:

```
<a href="https://studiolab.sagemaker.aws/import/github/org/repo/blob/master/path/to/notebook.ipynb">
  <img src="https://studiolab.sagemaker.aws/studiolab.svg" alt="Open In SageMaker Studio Lab"/>
</a>
```

This will creates a badge like:

[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/d2l-ai/d2l-pytorch-sagemaker-studio-lab/blob/161e45f1055654c547ffe3c81bd5f06310e96cff/GettingStarted-D2L.ipynb)