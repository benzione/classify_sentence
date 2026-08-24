**Assignment Overview**  
A new system is intended to get a user query in natural language and translate it into system’s API request (in JSON format) efficiently,   
To do so, one of the most important tasks is to correctly identify the relevant entities.  
Your task is to build an optimized ML pipeline for entity extraction that can perform on limited resources (CPU or local GPU).  
   
**Dataset Description**  
Two CSV files are provided to assist you in this task:

1. **user\_queries.csv**:  
   * **Columns**:  
     * **questions**: Contains the user’s textual query (e.g., "Show me all reports related to malware infection").  
     * **json**: Contains the corresponding JSON object that represents API query that matches the user question. Inside the Json object there are the relevant entities. The entities can be found in the ‘entityType’ key and, if present, also in the ‘relationTargetType’ key (which are part of the json object).  
2. **fields\_description.csv**:  
   * **Purpose**: This file provides details for each type of entity, including the relevant fields associated with the entity and a textual description of each field. While this information can help enhance the accuracy of entity prediction, it is not mandatory to use it.

   
**Objectives**

1. **ML Pipeline Design: Create an ML pipeline that:**  
   * **Extracts relevant entities from user queries based on input data.**  
   * **Optimizes resource usage to run efficiently on CPU or local GPU.**  
2. **Model Optimization:**  
   * **Incorporate optimization techniques.**  
   * **To align with our real-time assistance goals, your ML pipeline should be designed for low-latency processing, ensuring quick responses for each user query.**  
   * **Document the impact of each optimization step on model size, speed, and accuracy.**  
3. **Evaluation:**  
   * **Use metrics to evaluate entity extraction accuracy and efficiency.**  
   * **Provide examples of test cases where the pipeline performs well and where improvements are needed.**  
      

You should **Run the Entire Pipeline Locally.** Ensure that all components of the pipeline are executed on a local machine.  
\* The entities can be found in the entityType key and, if present, also in the relationTargetType key of the JSON column   
Please see below Example of expected input question/query,  json & expected output:  
 

| Use query (input) | Json | Expected output |
| :---- | :---- | :---- |
| What SMS messages were sent from suspicious phones to 0549876543 containing the word 'urgent'? | {'entityType': 'CDR', 'statements': \[{'type': 'filter', 'parameters': {'name': 'ifc.ootb.CDR.msisdn2', 'operator': 'equals', 'value': '0549876543'}}, {'type': 'filter', 'parameters': {'name': 'ifc.ootb.CDR.smsText', 'operator': 'contains', 'value': 'urgent'}}, {'type': 'filter', 'parameters': {'name': 'ifc.ootb.CDR.type', 'operator': 'equals', 'value': 'Text'}}, {'type': 'relation', 'parameters': {'relationType': \['relation\_Caller'\], 'relationTargetType': \['Phone'\]}, 'statements': \[{'type': 'filter', 'parameters': {'name': 'ifc.ootb.Participant.isSuspicious', 'operator': 'equals', 'value': True}}\]}\]}   | \[‘'CDR', ‘'Phone'’\] |

   
   
**Deliverables**

1. A Python script or Jupyter notebook with your implementatio  
2. A brief explanation of the approach you used, the techniques applied, and the justifications for your choices, including the metrics used to evaluate the performance.  
3. **Test Cases:** Run your pipeline on several test queries and provide the predicted entities.  
4. Open issues and how you further suggestions for further improvements

