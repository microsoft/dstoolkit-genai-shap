# SHAP (SHapley Additive exPlanations) for Generative AI Solutions


## Getting started

### Prerrequisites
- Install python 3.9.13: [https://www.python.org/downloads/release/python-3913/](https://www.python.org/downloads/release/python-3913/)

### Installation steps

1. Clone this repo locally: [https://github.com/microsoft/dstoolkit-genai-shap](https://github.com/microsoft/dstoolkit-genai-shap)
   > `git clone https://github.com/microsoft/dstoolkit-genai-shap.git`
2. Create virtual environment and activate it:
   > `cd dstoolkit-genai-shap.git`
   > `python3 -m venv .venv`
3. Define the encoding setting for the virtual environment:
   * For Windows:
     > Navigate to `.venv\Scripts\ctivate.bat`
     > Add these two lines at the bottom of the file:
      > `set PYTHONIOENCODING=utf-8`
      > `set PYTHONUTF8=1`

4. Install package dependencies to run the notebooks:
   > `pip install -r requirements.txt`
5. Create `.env` file by copying the `.env.template` file:
   > `cp .env.template .env`
6. Edit `.env` file and update the environment variables.
7. Open jupyter lab:
   > `jupyter lab`
8. Execute the following notebooks under the `docs/examples` folder and follow the steps:
   * `01-create-test-dataset.ipynb`
   * `02-gaishap-featurizer.ipynb`
   * `03-gaishap-blackbox-model.ipynb`


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
