<h1 align="center">
    <img 
        src="https://i.imgur.com/MpoaCk8.jpg" 
        width="200" 
        border="1" />
</h1>
<h1 align="center">
    <b>GenAISHAP</b>
</h1>
<h4 align="center">
    <i>Explainations for Generative AI, LLM-and-SLM-Based, Solutions</i> ⚡️
</h4>

---

Generative AI SHAP (GenAISHAP) is a python library that supports the creation of explanations to the metrics obtained for solutions based on LLMs (Large Language Models) or SLMs (Small Language Models). 

When building an LLM-or-SLM based solution one of the first challenges is how to measure the quality of the responses from the "Agent". Here, libraries like  [RAGAS](https://github.com/explodinggradients/ragas) or [promptflow](https://github.com/microsoft/promptflow) help on the evaluation of the quality of the solution by using metrics like **faithfulness**, **groundedness**, **context precision**, **context recall**, among others.

The next challenge is to add explainability to those quality metrics.  To answer questions like: 

- *Why a particular question is marked with a higher/lower metric (e.g., faithfulness)?*
- *What are the common characteristics of the user questions that produce good or bad **faithfulness**?*
- *What type of prompts produce better or lower **context recall**?*

The answer of those questions helps on the debugging of the overall solution and gives more insights to where to focus the next steps to improve the metrics.

***GenAISHAP*** was created to fulfill that need. It works as follows.

***GenAISHAP*** will create regression models, which we call them **black-box models**, for each of the metrics and will use those black-box models to produce explanations for each metric. The models are created from features extracted from the provided questions. Those **question features** could be generated automatically, using a tool, named **Featurizer** incorporated in the library or they can be manually created.

### Input

***GenAISHAP*** starts with a simple evaluation dataset with at least the following columns:

- User input, question or prompt. The column name should be `user_input` and its type should be string.
- One column for each metric already calculated, for example **faithfulness**, **context precision**, **context recall**.  All numerical columns will be assumed to be a metric column.

Example of the input using the [Mini Esg Bench Dataset](https://llamahub.ai/l/llama_datasets/Mini%20ESG%20Bench%20Dataset?from=llama_datasets):

```python
import pandas as pd

df_test_dataset = pd.read_json('./test-dataset.json', orient='records')
df_test_dataset.head(10)
```

<img src="https://i.imgur.com/LFqcvM7.png" width="1200" />

> In this example, the column `user_input` will be used to refer to the user prompt, and the columns `faithfulness`, `context_precision` and `context_recall` will be used as metric columns since those columns are numerical.
>
> The other columns, `retrieved_contexts`, `response`, and `reference` are not needed for **GenAISHAP** but are normally required for the calculation of the metrics.

### Featurizer

***GenAISHAP*** has an utilily to automatically create features from the `user_input` entries.  Those features, are characteristics of the user questions that will be used as regressors to train a black-box model that will be used to calculate the explanations.  It is possible to manually add, remove, or modify those automatically generated features to improve the quality of the explanations.

The following is an example of how to automatically generate the features that will be used as regressors for the black-box model:

```python
from genaishap import Featurizer

# Loads the input from the df_test_dataset pandas DataFrame
featurizer = Featurizer.from_pandas(df_test_dataset)

# Creates the features automatically using only the user_input column
featurizer.create_features_using_azure_openai(deployment_name="gpt-4o", num_features=12)
```

The following are the features generated:

- there_is_any_company_identified_in_the_question
- list_of_companies_identified_in_the_question
- there_is_any_initiative_or_program_identified_in_the_question
- list_of_initiatives_or_programs_identified_in_the_question
- there_is_any_financial_or_environmental_metric_identified_in_the_question
- list_of_financial_or_environmental_metrics_identified_in_the_question
- there_is_any_specific_year_identified_in_the_question
- list_of_years_identified_in_the_question
- is_a_question_about_trends_or_changes_over_time
- is_a_question_about_identification_of_factors_or_insights
- is_a_question_related_to_a_specific_page_or_section_of_a_document
- is_a_question_about_goals_or_targets

> Currently, there are two types of features supported: **boolean** and **list of strings**. The goal is to be able to capture the characteristics of the different user queries in a way that can be easily interpretable by a human, and at the same time these features should be able to be engineered to be used as regressors for the black-box regression models.

Then, ***GenAISHAP*** also includes another utility to automatically fill out the values for each user input for each feature. 

```python
featurizer.fill_out_features_using_azure_openai(deployment_name="gpt-4o", batch_size=20)
```

> The batch size is used to control how many user inputs will be filled out per LLM call, to avoid overflow of the total number of tokens of the deployed model used.

A sample of the output is the following:

```python
df_features = featurizer.to_pandas()
```

<img src="https://i.imgur.com/Heevl9i.png" width="1200" />

### Explainers

Once, the user input features and its corresponding values are defined, ***GenAISHAP*** can start working on the next steps:

- Feature engineering
- Regression black-box model training
- Creation of SHAP explainers

```python
from genaishap import GenAIExplainer

gai_explainer = GenAIExplainer.from_pandas(df_test_dataset, df_features)

# Feature Engineering
gai_explainer.feature_engineering()

# Train black-box model and create explainers
gai_explainer.create_explainers()

print(gai_explainer.r2_scores_)
```

The following are examples of the coefficients of determination (r2) scores for each of the best model trained for each metric:

- **faithfulness**: 0.734
- **context_precision**: 0.846
- **context_recall**: 0.820

> During the training and selection of the best models a **t-test** is performed to evaluate if the estimated metric using the models produces a statistically related sample from the same population of the original metric: fail to reject the null hypothesis that both, the original metric and the estimated metric are samples from the same population. If the t-test rejects the null hypothesis a warning message is displayed during the creation of the explainers.  The explainers cannot be used as reference. 

> Also, as a rule of thumb, if the `r2_score` is high (>0.75) the explanations of black-box model could be used as reference. If it is lower, like the r2-score of the **faithfulness** in this example, the use of the explainers could produce misleading conclusions.

Just as an example, let's use **context recall** for now.  It is possible, at this point to generate explanations at the full dataset level to answer questions like: *What are the more relevant features of the questions that drives a higher, or lower **context recall**?*

It is possible to do it, for example, using the SHAP summary plot, as follows:

```python
metric = 'context_recall'

X = pd.DataFrame(gai_explainer.preprocessed_features)
metric_explainer = gai_explainer.explainers_[metric]
shap_values = metric_explainer(X)

shap.summary_plot(shap_values, X, plot_size=(20,10))
```

The following is the SHAP Summary Plot generated:

<img src="https://i.imgur.com/oMXDJkq.png" width="1200" />

> From this plot we can conclude, for example, that the **context recall**:
> - Is higher when there is a company identified in the question.
> - Also is higher when the question is about goals about goals and targets.
> - Also, we can conclude that the **context recall** is lower in the questions with explicit reference to Amazon, in contrast with the questions with explicit reference to Apple or Facebook. 
> 
> This type of information can be used as insights to guide next steps to improve the overall **context recall** of the solution.

During the creation of the explainers other warnings related to the safe use of the explainers can be raised. For example warinings like the following can be rised when creating the explainers for the **context recall** metric:

> `UserWarning: There are 6 estimated values in the metric context_recall far from the original values. The following is the list of indexes [24, 25, 26, 27, 31, 41].`

These warnings are shown because during the creation of the training of the black-box model to create the explainers, there is a process to evaluate how far are the estimated values of each metric compared to the original one, using t-distribution and confidence intervals.  If an instance is out of the confidence interval it is marked as too far from the original value and the warning is shown to alert the user to use carefully the instance explanations for those specific instances.

The following table shows a comparison of the original metric values compared with the estimated values calculated using the black-box model, and the identification if the instance is **out of range** and therefore the explanations should be used carefully for those instances.

<img src="https://i.imgur.com/tWL7DHG.png" width="300" />

This type of table can be generated using a code like:

```python
df_metric = pd.DataFrame(gai_explainer.metrics)[[metric]]
df_metric['estimated_value'] = gai_explainer.estimators_[metric].predict(X)
df_metric['is_out_of_range'] = gai_explainer.is_out_of_range_[metric]

df_metric.style.apply(
    lambda s : [
        'background-color: yellow' if s.loc['is_out_of_range'] else '' for v in s.index
    ], 
    axis=1
)
```

As an example let's pick the **14th index**, which has a context recall of 0, and an extimated value of 0.001. The following are the details of that instance:


### INDEX 14

**USER INPUT:**
What household brands were featured in the in the climate pledge infographic on page 14?

**RETRIEVED CONTEXT:**


**CHUNK 1:**

Customer Shopping Experience We have enabled 19 shopping features, such as new search functions and clearer digital badges, to highlight Climate Pledge Friendly products. These features have helped Amazon customers switch to a Climate Pledge Friendly product— something that occurs the first time customers purchase a Climate Pledge Friendly product in a category for which they have only purchased non-Climate Pledge Friendly products over the prior two-year period. Customers can now shop for products across Apparel, Home, Electronics, and Kitchen categories, meaning customers now have access to 550,000 Climate Pledge Friendly products, up from 250,000 in 2021. We have also seen strong adoption of the Climate Pledge Friendly program from Amazon Business customers. Amazon Business helps companies create guided buying policies, which place preference on sustainable products qualified through the Climate Pledge Friendly program. As of December 2022, 18,000 businesses had these buying policies in place. Climate Pledge Friendly product badge on Amazon.com. 53 2022 Amazon Sustainability Report Introduction PeopleAppendixSustainability Innovating Our Products and Services | Product Sustainability O2

**CHUNK 2:**

As of December 2022, 18,000 businesses had these buying policies in place. Looking Forward We are committed to continually improving product sustainability, enhancing responsible materials and commodities sourcing, and collaborating with supply chain partners to drive adoption of new regulations and compliance requirements. To promote responsible sourcing practices more widely, we will continue to advocate for robust standards that limit negative social and environmental impacts. We will also continue to improve device efficiency while delivering new and better ways for customers to shop for more- sustainable products. Climate Pledge Friendly product badge on Amazon.com. Sustainable
Product Selection We aim to provide customers with more-sustainable alternatives throughout their shopping experience, offering them products with improvements in at least one sustainability area. Amazon’s flagship sustainable shopping program, Climate Pledge Friendly, is how we do this. Looking Forward We are committed to continually improving product sustainability, enhancing responsible materials and commodities sourcing, and collaborating with supply chain partners to drive adoption of new regulations and compliance requirements. To promote responsible sourcing practices more widely, we will continue to advocate for robust standards that limit negative social and environmental impacts. We will also continue to improve device efficiency while delivering new and better ways for customers to shop for more- sustainable products. Climate Pledge Friendly Product Certification As of December 2022, the Climate Pledge Friendly program added 16 certifications, giving selling partners 52 ways to qualify for Climate Pledge Friendly. The new certifications recognize improvements in at least one aspect of sustainability, from recycled content to energy efficiency. This year’s additions include, but are not limited to: • EU Energy Label grades A and B • U.S. Environmental Protection Agency (EPA) Design for the Environment and WaterSense • Fairtrade International • NATRUE • STANDARD 100 by OEKO-TEX • Business + Institutional Furniture Manufacturers Association Level • GreenCircle Device Certifications We partner with trusted, transparent external certifications to validate the sustainability of our products and clearly communicate this to customers through the Climate Pledge Friendly badge. Since 2020, many of our Echo, Fire TV, Fire tablet, Kindle e-reader, and smart home devices and accessories have received sustainability certifications included in the Climate Pledge Friendly program. Many of these products qualified for the Climate Pledge Friendly badge by achieving the Carbon Trust’s Reducing C certification, which highlights products with an associated carbon footprint that is decreasing annually. The Amazon Smart Thermostat is the first Amazon device to be ECOLOGO Silver certified, demonstrating it meets standards for reducing environmental impacts at one or more product lifecycle stages. To further reduce the thermostat’s environmental impact, we have also now introduced a new carbon-emissions-optimization feature. By monitoring local grid emissions data in real time, the thermostat can automatically adjust set temperature points to reduce energy use during high-emission periods, such as when grids are using less sustainable power sources.  Learn more about device energy efficiency . Climate Pledge Friendly also highlights products designed for circularity. Today, customers can choose from 20,000 refurbished products through Pre-Owned Certified, a new certification recognizing products that are inspected, cleaned, and repaired to excellent functional standards. Pre-Owned Certified aims to extend the life of products, reducing e-waste and raw material extraction. The inclusion of Pre- Owned Certified highlights the value we place on product reuse and circularity, giving customers access to high-quality refurbished electronics. Customer Shopping Experience We have enabled 19 shopping features, such as new search functions and clearer digital badges, to highlight Climate Pledge Friendly products. These features have helped Amazon customers switch to a Climate Pledge Friendly product— something that occurs the first time customers purchase a Climate Pledge Friendly product in a category for which they have only purchased non-Climate Pledge Friendly products over the prior two-year period. Customers can now shop for products across Apparel, Home, Electronics, and Kitchen categories, meaning customers now have access to 550,000 Climate Pledge Friendly products, up from 250,000 in 2021. We have also seen strong adoption of the Climate Pledge Friendly program from Amazon Business customers. Amazon Business helps companies create guided buying policies, which place preference on sustainable products qualified through the Climate Pledge Friendly program.

**RESPONSE:**
The context does not provide information about household brands featured in the climate pledge infographic on page 14.

**REFERENCE:**
The initial idea for a startup by Paul Graham and Robert Morris was to put art galleries online. The idea failed because art galleries didn't want to be online, especially the fancy ones, as that's not how they sell. They wrote software to generate web sites for galleries and to resize images and set up an http server to serve the pages, but they struggled to sign up galleries. Even when they offered to make sites for free, they couldn't get galleries to pay for the service.

**METRIC:** context_recall

**METRIC Value:** 0.000

**MODEL ESTIMATED Value:** -0.000

```python
index = 14
shap.waterfall_plot(shap_values[index])
```

The generated plot is:

<img src="https://i.imgur.com/RBmYBJC.png" width="800" />

> The horizontal axis of the SHAP waterfall plot shows the contributions of individual features to the model's prediction for a specific instance. The sum of all the contributions will be the final instance predicted value. 
>
> From this plot we can conclude that the reduction of the **context recall** for this specific instance was mainly driven by don't having an explicit mention of a company together with the mention of an specific initiative or program
> 
> This type of information adds insights at the instance level on how to improve the overall quality of the solution.

## Example Notebooks

The following are the steps to be able to execute the example notebooks

### Prerrequisites
- Install docker desktop: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- [Only for Windows] Install Make for Windows: [https://gnuwin32.sourceforge.net/packages/make.htm](https://gnuwin32.sourceforge.net/packages/make.htm)

### Installation steps

1. Clone this repo locally: [https://github.com/microsoft/dstoolkit-genai-shap](https://github.com/microsoft/dstoolkit-genai-shap)
   > `git clone https://github.com/microsoft/dstoolkit-genai-shap.git`
2. Build GenAISHAP image:
   > `cd dstoolkit-genai-shap`
   > 
   > `make build-image`
3. Create `.env` file by copying the `.env.template` file:
   > `cp .env.template .env`
4. Edit `.env` file and update the environment variables.
5. Run GenAISHAP container and open jupyter lab url:
   > `make run-container`
   > 
   > Open the jupyter lab url by copying the line in the log that looks like:
   > 
   > `http://127.0.0.1:8888/lab?token=8a3eb1ebf39038598e0b6ce7cc400bf841b2b3891998ceb4`
   > 
   > and open it in a local web navigator like Microsoft Edge.
6. Execute the following notebooks under the `docs/examples` folder and follow the steps:
   * `01-create-test-dataset.ipynb`
   * `02-genaishap-featurizer.ipynb`
   * `03-genaishap-explainers.ipynb`


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
