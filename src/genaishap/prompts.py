""" Library with all the prompt templates
"""

# TODO: Evaluate few shot vs one shot performance
_CREATE_FEATURES_PROMPT : str = """
Use the following questions (user inputs) to create a list of {num_features} features to classify each question. 

- Use JSON format to present the output.
- For each feature define its type among the following: "boolean" or "list_of_strings"

---

# Example 1

Producing 12 features from 9 questions.

## Input:

[QUESTION 1] How has Apple's total net sales changed over time?
[QUESTION 2] What are the major factors contributing to the change in Apple's gross margin in the most recent 10-Q compared to the previous quarters?
[QUESTION 3] Has there been any significant change in Apple's operating expenses over the reported quarters? If so, what are the key drivers for this change?
[QUESTION 4] How has Apple's revenue from iPhone sales fluctuated across quarters?
[QUESTION 5] Can any trends be identified in Apple's Services segment revenue over the reported periods?
[QUESTION 6] What is the impact of foreign exchange rates on Apple's financial performance? List this out separately for each reported period.
[QUESTION 7] Are there any notable changes in Apple's liquidity position or cash flows as reported in these 10-Qs?
[QUESTION 8] Examine how Intel's effective tax rate in the most recent 10-Q compares with the tax-related discussions in the notes section.
[QUESTION 9] In Amazon's latest 10-Q, how does the revenue distribution across its diverse business segments like e-commerce, AWS, and others compare to the costs incurred in these segments?

## Output:

{{
    "features": [
        {{"feature": "there_is_any_company_indentified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_companies_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_business_segment_indentified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_business_segments_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_financial_metric_identified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_if_financial_metrics_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_quarters_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "is_a_comparison_question", "ftype": "boolean"}},
        {{"feature": "is_a_question_about_trends_or_changes_over_time", "ftype": "boolean"}},
        {{"feature": "is_a_question_about_identification_of_factors_or_insights", "ftype": "boolean"}},
        {{"feature": "is_a_question_related_to_the_notes_section", "ftype": "boolean"}},
    ]
}}

---

The following are the questions for you to process:

## Input:

{questions}

## Output:

"""

# TODO: Evaluate few shot vs one shot performance
_FILL_OUT_FEATURES_PROMPT : str = """
Use the following list of questions as source to fill out each feature.

- Use JSON format to present the output.
- The definition of the features is in JSON format.
- Each feature has a type from the following list of feature types: "boolean" or "list_of_strings"

---

# Example 1

## Input:

[FEATURES]

{{
    "features": [
        {{"feature": "there_is_any_company_indentified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_companies_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_business_segment_indentified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_business_segments_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_financial_metric_identified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_if_financial_metrics_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "ftype": "boolean"}},
        {{"feature": "list_of_quarters_identified_in_the_question", "ftype": "list_of_strings"}},
        {{"feature": "is_a_comparison_question", "ftype": "boolean"}},
        {{"feature": "is_a_question_about_trends_or_changes_over_time", "ftype": "boolean"}},
        {{"feature": "is_a_question_about_identification_of_factors_or_insights", "ftype": "boolean"}},
        {{"feature": "is_a_question_related_to_the_notes_section", "ftype": "boolean"}},
    ]
}}

[QUESTION 1] How has Apple's total net sales changed over time?
[QUESTION 2] What are the major factors contributing to the change in Apple's gross margin in the most recent 10-Q compared to the previous quarters?
[QUESTION 3] Has there been any significant change in Apple's operating expenses over the reported quarters? If so, what are the key drivers for this change?
[QUESTION 4] How has Apple's revenue from iPhone sales fluctuated across quarters?
[QUESTION 5] Can any trends be identified in Apple's Services segment revenue over the reported periods?
[QUESTION 6] What is the impact of foreign exchange rates on Apple's financial performance? List this out separately for each reported period.
[QUESTION 7] Are there any notable changes in Apple's liquidity position or cash flows as reported in these 10-Qs?
[QUESTION 8] Examine how Intel's effective tax rate in the most recent 10-Q compares with the tax-related discussions in the notes section.
[QUESTION 9] In Amazon's latest 10-Q, how does the revenue distribution across its diverse business segments like e-commerce, AWS, and others compare to the costs incurred in these segments?

## Output:

{{
    "feature_values": [
        {{
            "question": "How has Apple's total net sales changed over time?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["total net sales"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": false}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": []}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": true}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "What are the major factors contributing to the change in Apple's gross margin in the most recent 10-Q compared to the previous quarters?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["gross margin"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": true}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": ["recent 10-Q"]}},
                {{"feature": "is_a_comparison_question", "value": true}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": false}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": true}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "Has there been any significant change in Apple's operating expenses over the reported quarters? If so, what are the key drivers for this change?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["operating expenses"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": false}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": []}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": true}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": true}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "How has Apple's revenue from iPhone sales fluctuated across quarters?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": ["iPhone"]}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["sales revenue"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": false}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": []}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": true}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "Can any trends be identified in Apple's Services segment revenue over the reported periods?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": ["Services segment"]}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["revenue"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": false}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": []}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": true}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "What is the impact of foreign exchange rates on Apple's financial performance? List this out separately for each reported period."
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["foreign exchange rates"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": false}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": []}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": false}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": true}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "Are there any notable changes in Apple's liquidity position or cash flows as reported in these 10-Qs?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Apple"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["liquidity position", "cash flows"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": true}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": ["10-Qs"]}},
                {{"feature": "is_a_comparison_question", "value": false}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": true}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
        {{
            "question": "Examine how Intel's effective tax rate in the most recent 10-Q compares with the tax-related discussions in the notes section."
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Intel"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": false}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": []}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["tax rate"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": true}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": ["10-Qs"]}},
                {{"feature": "is_a_comparison_question", "value": true}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": false}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": true}},
            ]
        }},
        {{
            "question": "In Amazon's latest 10-Q, how does the revenue distribution across its diverse business segments like e-commerce, AWS, and others compare to the costs incurred in these segments?"
            "features": [
                {{"feature": "there_is_any_company_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_companies_identified_in_the_question", "value": ["Amazon"]}},
                {{"feature": "there_is_any_business_segment_indentified_in_the_question", "value": true}},
                {{"feature": "list_of_business_segments_identified_in_the_question", "value": ["e-commerce", "AWS"]}},
                {{"feature": "there_is_any_financial_metric_identified_in_the_question", "value": true}},
                {{"feature": "list_if_financial_metrics_identified_in_the_question", "value": ["revenue", "costs"]}},
                {{"feature": "there_is_any_specific_quarter_identified_in_the_question", "value": true}},
                {{"feature": "list_of_quarters_identified_in_the_question", "value": ["10-Qs"]}},
                {{"feature": "is_a_comparison_question", "value": true}},
                {{"feature": "is_a_question_about_trends_or_changes_over_time", "value": false}},
                {{"feature": "is_a_question_about_identification_of_factors_or_insights", "value": false}},
                {{"feature": "is_a_question_related_to_the_notes_section", "value": false}},
            ]
        }},
    ]
}}

---

The following is the input for you to process:

## Input:

[FEATURES]

{features}

{question_batch}

## Output:

"""