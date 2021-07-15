![GDDS](https://user-images.githubusercontent.com/51120437/125526537-c57d2eef-414f-4ac2-bb59-06a347f3c386.png)

# Grid Search With Decision Tree
### To understand the code you must learn (Decision tree algorithm, ROC, AUC, Grid_searcher, predict_proba)

#### We will use the dataset from a Kaggle machine learning competition Click-Through Rate Prediction (https://www.kaggle.com/c/avazu-ctr-prediction).

## The data fields are described as follows:
1-id: ad identifier, such as 1000009418151094273, 10000169349117863715.

2-click: 0 for non-click, 1 for click.

3-hour: in the format of YYMMDDHH, for example, 14102100.

4-C1: anonymized categorical variable, such as 1005, 1002.

5-banner_pos: where a banner is located, 1 and 0.

6-site_id: site identifier, such as 1fbe01fe, fe8cc448, d6137915.

7-site_domain: hashed site domain, such as 'bb1ef334', 'f3845767.

8-site_category: hashed site category, such as 28905ebd, 28905ebd.

9-app_id: mobile app identifier.

10-app_domain.

11-app_category.

12-device_id: mobile device identifier.

13-device_ip: IP address.

14-device_model: such as iPhone 6, Samsung, hashed by the way 

15-device_type: such as tablet, smartphone, hashed device_conn_type.

16-Wi-Fi or 3G for example, again hashed in the data.

17-C14-C21: anonymized categorical variables.


### Don't be scared by the anonymized and hashed values. They are categorical features and each possible value of them corresponds to a real and meaningful value, but it is presented this way due to the 
# privacy policy. Maybe C1 means user gender, and 1005 and 1002 represent male and female respectively.
