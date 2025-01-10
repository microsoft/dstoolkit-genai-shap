<h1 align="center">
    <img 
        src="./docs/img/logo_white_bg.jpeg" 
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

Example of the input using the [Paul Graham Essay Dataset](https://llamahub.ai/l/llama_datasets/Paul%20Graham%20Essay?from=llama_datasets):

```python
import pandas as pd

df_test_dataset = pd.read_json('./test-dataset.json', orient='records')
df_test_dataset.head(10)
```

<img src="./docs/img/input_example.png" width="1200" />

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

- there_is_any_person_identified_in_the_question
- list_of_people_identified_in_the_question
- there_is_any_technology_or_tool_identified_in_the_question
- list_of_technologies_or_tools_identified_in_the_question
- there_is_any_academic_or_professional_transition_identified_in_the_question
- list_of_academic_or_professional_transitions_identified_in_the_question
- there_is_any_company_or_organization_identified_in_the_question
- list_of_companies_or_organizations_identified_in_the_question
- is_a_question_about_personal_experiences_or_decisions
- is_a_question_about_technological_or_software_development
- is_a_question_about_artistic_or_creative_processes
- is_a_question_related_to_business_or_startup_strategies

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

<img src="./docs/img/featurizer_example.png" width="1200" />

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

- **faithfulness**: 0.752
- **context_precision**: 0.999
- **context_recall**: 0.938

> During the training and selection of the best models a **t-test** is performed to evaluate if the estimated metric using the models produces a statistically related sample from the same population of the original metric: fail to reject the null hypothesis that both, the original metric and the estimated metric are samples from the same population. If the t-test rejects the null hypothesis a warning message is displayed during the creation of the explainers.  The explainers cannot be used as reference. 

> Also, as a rule of thumb, if the `r2_score` is high (>0.75) the explanations of black-box model could be used as reference. If it is lower, the use of the explainers could produce misleading conclusions.

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

<img src="./docs/img/shap_summary_plot_example.png" width="1200" />

> From this plot we can conclude, for example, that the **context recall**:
> - Is higher when the question is not about personal experiences or decisions.
> - In contrast, questions where it is possible to identify a company or there is a explicit reference to the author have higher **context recall**.
> 
> This type of information can be used as insights to guide next steps to improve the overall **context recall** of the solution.

During the creation of the explainers other warnings related to the safe use of the explainers can be raised. For example warinings like the following can be rised when creating the explainers for the **context recall** metric:

> `UserWarning: There are 6 estimated values in the metric context_recall too far from the original values. The following is the list of indexes [18, 19, 26, 27, 37, 39].`

These warnings are shown because during the creation of the training of the black-box model to create the explainers, there is a process to evaluate how far are the estimated values of each metric compared to the original one, using t-distribution and confidence intervals.  If an instance is out of the confidence interval it is marked as too far from the original value and the warning is shown to alert the user to use carefully the instance explanations for those specific instances.

The following table shows a comparison of the original metric values compared with the estimated values calculated using the black-box model, and the identification if the instance is **out of range** and therefore the explanations should be used carefully for those instances.

<img src="./docs/img/out_of_range_example.png" width="300" />

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
In the essay by Paul Graham, what was the initial idea for a startup that he and Robert Morris had, and why did it fail?

**RETRIEVED CONTEXT:**


**CHUNK 1:**

Over the next several years I wrote lots of essays about all kinds of different topics. O'Reilly reprinted a collection of them as a book, called Hackers & Painters after one of the essays in it. I also worked on spam filters, and did some more painting. I used to have dinners for a group of friends every thursday night, which taught me how to cook for groups. And I bought another building in Cambridge, a former candy factory (and later, twas said, porn studio), to use as an office.

One night in October 2003 there was a big party at my house. It was a clever idea of my friend Maria Daniels, who was one of the thursday diners. Three separate hosts would all invite their friends to one party. So for every guest, two thirds of the other guests would be people they didn't know but would probably like. One of the guests was someone I didn't know but would turn out to like a lot: a woman called Jessica Livingston. A couple days later I asked her out.

Jessica was in charge of marketing at a Boston investment bank. This bank thought it understood startups, but over the next year, as she met friends of mine from the startup world, she was surprised how different reality was. And how colorful their stories were. So she decided to compile a book of interviews with startup founders.

When the bank had financial problems and she had to fire half her staff, she started looking for a new job. In early 2005 she interviewed for a marketing job at a Boston VC firm. It took them weeks to make up their minds, and during this time I started telling her about all the things that needed to be fixed about venture capital. They should make a larger number of smaller investments instead of a handful of giant ones, they should be funding younger, more technical founders instead of MBAs, they should let the founders remain as CEO, and so on.

One of my tricks for writing essays had always been to give talks. The prospect of having to stand up in front of a group of people and tell them something that won't waste their time is a great spur to the imagination. When the Harvard Computer Society, the undergrad computer club, asked me to give a talk, I decided I would tell them how to start a startup. Maybe they'd be able to avoid the worst of the mistakes we'd made.

So I gave this talk, in the course of which I told them that the best sources of seed funding were successful startup founders, because then they'd be sources of advice too. Whereupon it seemed they were all looking expectantly at me. Horrified at the prospect of having my inbox flooded by business plans (if I'd only known), I blurted out "But not me!" and went on with the talk. But afterward it occurred to me that I should really stop procrastinating about angel investing. I'd been meaning to since Yahoo bought us, and now it was 7 years later and I still hadn't done one angel investment.

Meanwhile I had been scheming with Robert and Trevor about projects we could work on together. I missed working with them, and it seemed like there had to be something we could collaborate on.

As Jessica and I were walking home from dinner on March 11, at the corner of Garden and Walker streets, these three threads converged. Screw the VCs who were taking so long to make up their minds. We'd start our own investment firm and actually implement the ideas we'd been talking about. I'd fund it, and Jessica could quit her job and work for it, and we'd get Robert and Trevor as partners too. [13]

Once again, ignorance worked in our favor. We had no idea how to be angel investors, and in Boston in 2005 there were no Ron Conways to learn from. So we just made what seemed like the obvious choices, and some of the things we did turned out to be novel.

There are multiple components to Y Combinator, and we didn't figure them all out at once. The part we got first was to be an angel firm. In those days, those two words didn't go together. There were VC firms, which were organized companies with people whose job it was to make investments, but they only did big, million dollar investments. And there were angels, who did smaller investments, but these were individuals who were usually focused on other things and made investments on the side. And neither of them helped founders enough in the beginning. We knew how helpless founders were in some respects, because we remembered how helpless we'd been. For example, one thing Julian had done for us that seemed to us like magic was to get us set up as a company.

**CHUNK 2:**

So we just made what seemed like the obvious choices, and some of the things we did turned out to be novel.

There are multiple components to Y Combinator, and we didn't figure them all out at once. The part we got first was to be an angel firm. In those days, those two words didn't go together. There were VC firms, which were organized companies with people whose job it was to make investments, but they only did big, million dollar investments. And there were angels, who did smaller investments, but these were individuals who were usually focused on other things and made investments on the side. And neither of them helped founders enough in the beginning. We knew how helpless founders were in some respects, because we remembered how helpless we'd been. For example, one thing Julian had done for us that seemed to us like magic was to get us set up as a company. We were fine writing fairly difficult software, but actually getting incorporated, with bylaws and stock and all that stuff, how on earth did you do that? Our plan was not only to make seed investments, but to do for startups everything Julian had done for us.

YC was not organized as a fund. It was cheap enough to run that we funded it with our own money. That went right by 99% of readers, but professional investors are thinking "Wow, that means they got all the returns." But once again, this was not due to any particular insight on our part. We didn't know how VC firms were organized. It never occurred to us to try to raise a fund, and if it had, we wouldn't have known where to start. [14]

The most distinctive thing about YC is the batch model: to fund a bunch of startups all at once, twice a year, and then to spend three months focusing intensively on trying to help them. That part we discovered by accident, not merely implicitly but explicitly due to our ignorance about investing. We needed to get experience as investors. What better way, we thought, than to fund a whole bunch of startups at once? We knew undergrads got temporary jobs at tech companies during the summer. Why not organize a summer program where they'd start startups instead? We wouldn't feel guilty for being in a sense fake investors, because they would in a similar sense be fake founders. So while we probably wouldn't make much money out of it, we'd at least get to practice being investors on them, and they for their part would probably have a more interesting summer than they would working at Microsoft.

We'd use the building I owned in Cambridge as our headquarters. We'd all have dinner there once a week â on tuesdays, since I was already cooking for the thursday diners on thursdays â and after dinner we'd bring in experts on startups to give talks.

We knew undergrads were deciding then about summer jobs, so in a matter of days we cooked up something we called the Summer Founders Program, and I posted an announcement on my site, inviting undergrads to apply. I had never imagined that writing essays would be a way to get "deal flow," as investors call it, but it turned out to be the perfect source. [15] We got 225 applications for the Summer Founders Program, and we were surprised to find that a lot of them were from people who'd already graduated, or were about to that spring. Already this SFP thing was starting to feel more serious than we'd intended.

We invited about 20 of the 225 groups to interview in person, and from those we picked 8 to fund. They were an impressive group. That first batch included reddit, Justin Kan and Emmett Shear, who went on to found Twitch, Aaron Swartz, who had already helped write the RSS spec and would a few years later become a martyr for open access, and Sam Altman, who would later become the second president of YC. I don't think it was entirely luck that the first batch was so good. You had to be pretty bold to sign up for a weird thing like the Summer Founders Program instead of a summer job at a legit place like Microsoft or Goldman Sachs.

The deal for startups was based on a combination of the deal we did with Julian ($10k for 10%) and what Robert said MIT grad students got for the summer ($6k). We invested $6k per founder, which in the typical two-founder case was $12k, in return for 6%. That had to be fair, because it was twice as good as the deal we ourselves had taken. Plus that first summer, which was really hot, Jessica brought the founders free air conditioners.

**RESPONSE:**
The essay does not provide specific details about an initial startup idea that Paul Graham and Robert Morris had, nor does it mention why such an idea might have failed.

**REFERENCE:**
The initial idea for a startup by Paul Graham and Robert Morris was to put art galleries online. The idea failed because art galleries didn't want to be online, especially the fancy ones, as that's not how they sell. They wrote software to generate web sites for galleries and to resize images and set up an http server to serve the pages, but they struggled to sign up galleries. Even when they offered to make sites for free, they couldn't get galleries to pay for the service.

**METRIC:** context_recall

**METRIC Value:** 0.000

**MODEL ESTIMATED Value:** 0.001

```python
index = 14
shap.waterfall_plot(shap_values[index])
```

The generated plot is:

<img src="./docs/img/waterfall_example.png" width="800" />

> The horizontal axis of the SHAP waterfall plot shows the contributions of individual features to the model's prediction for a specific instance. The sum of all the contributions will be the final instance predicted value. 
>
> From this plot we can conclude that the reduction of the **context recall** for this specific instance was mainly driven by the explicit mention of "Robert Morris" and because it is a question with no mention of any specific company in the question.  Looking at the details of this instance we can conclude that LlamaIndex was not able to pick the right documents that includes the answer to the question, and that the drivers of the error could be the combination of the mention of Robert Morris together with the absence of the mention of a specific company.
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
