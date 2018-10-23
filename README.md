# Text Mining on Armenian Online Job Postings 

**Dataset :** https://www.kaggle.com/udacity/armenian-online-job-postings/home

**Report :** https://github.com/lppier/Armenian_Online_Job_Postings_Text_Mining/blob/master/Text_Mining_Jobs_Posting_Report.pdf

**Techniques Used:** 
1. Text pre-processing
2. Regex
3. K-means Clustering
4. Topic Modelling
5. Text-based Classification
6. Information Extraction


Online job advertisements have become the dominant job searching and employer-employee job matching model in most developed economies around the world and gaining popularity in all parts of the world. It is estimated that in 2014 that more than 70 percent of job openings are posted online in the United States of America and by researching the detailed data of the online jobs ads, researchers can better understand the labour market.

This project analysed the online job ads posting from 2004 to 2015 posted on the CareerCenter.am, an Armenian online human resource centre. The main business objective is to understand the dynamics of the labour market of Armenia and relevant business questions were defined. A secondary objective is to implement advanced text analytics as a proof of concept to create classification and information retrieval function that can add additional value to the job portal.

We followed the CRISP-DM methodology for the project. After the business and data understanding with visualisation, we prepared the data. The Armenian job dataset was cleaned of missing data and duplicated sets. The pre-processing involves heavily the careful removal of non-essential characters such as newlines and punctuations and conversion of text to lower case. Also, depending on the task, tokenisation followed by lemmatisation is also done. Patterns that did not add much value such as emails and telephone numbers were also removed. We also expanded the stop-words dictionary for this task and other common words such as ‘Armenian’ which carries no additional information value.

By applying K-Means clustering, we created an understanding of the required qualifications and skillset in the Armenia labour market over the 10-year period from 2004 to 2015. IT related skills demands have shown to be on an increasing trend increased over the period. This increase and shift towards IT-related jobs are also validated by topic modelling that clearly shown that software development ‘topic’ has the strongest growth over the same period.

Applying supervised text mining techniques, we also demonstrated it is possible to create accurate classification models that create a filter out IT related job posting. We are also able to identify the type of companies that create job ads with our custom regex filters. In addition, we have shown that a job similarity search function is possible given the embeddings on the job ad text using cosine similarity between the vectors.

It is recommended that the Armenian government place more emphasis and effort to provide better education pathway for their citizens to gain IT skills to fulfil the IT skills labour demand. Likewise, for current and future job seekers, it is recommended that if they have interest, they should not hesitate to equip themselves with IT skillset, in particularly the IT operations and applications developments knowledge. In addition, the CareerCenter.am job portal website can utilise our project Information Extraction methods to improve the search capabilities and features of the website, enhancing the user experiences and capabilities of the job matching function.
