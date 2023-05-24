# ND

![image](https://github.com/kaburia/ND/assets/88529649/5ca461ab-3655-4142-9e2e-8a46a3d9a59b)
From the first run of the model based on the fit_summary()

The best model from fitting the train data to the Tabular Predictor is the "WeightedEnsemble_L3" model. Here are the details of the training run:

* The "WeightedEnsemble_L3" model achieved a score_val of -30.116344.
* The prediction time for the validation set (pred_time_val) was 23.465831 seconds.
* The fitting time (fit_time) for the model was 530.121128 seconds.
* The prediction time for the validation set (pred_time_val_marginal) was 0.000900 seconds.
* The fitting time for the model (fit_time_marginal) was 0.299811 seconds.
* The stack_level of the model is 3.
* The model can be used for inference (can_infer = True).
* The fit_order of the model is 14.

It's important to note that the "WeightedEnsemble_L3" model is a stacked ensemble model, which combines predictions from multiple base models to make the final predictions.<br>
It achieved the best performance among all the models trained in this run, as indicated by the lowest score_val.<br>

By adding the additional feature "hour" to the bike sharing dataset, the performance of the model improved, as indicated by the increase in the Kaggle score from 0.73149 to 0.71724.<br>
This improvement suggests that the "hour" feature contains valuable information that is relevant for predicting bike sharing demand.

The addition of the "hour" feature likely led to discoveries in the data that impacted the model performance. Here are some possible reasons for this improvement:

* Time-related patterns: The "hour" feature captures the specific hour of the day when bike sharing demand occurs. This information may help the model capture time-related patterns, such as peak hours or periods of high or low demand. By incorporating these patterns into the model, it can make more accurate predictions.

* User behavior: The "hour" feature can provide insights into user behavior and preferences. For example, people may have different tendencies to use bike sharing services during different times of the day. Understanding these patterns can help the model adjust its predictions accordingly.

* Interaction with other features: The addition of the "hour" feature may have introduced interactions with other features in the dataset. Certain combinations of features, such as the interaction between "hour" and "season" or "hour" and "weather," could have a significant impact on the bike sharing demand. The model can now capture these interactions and improve its predictive ability.

* Improved feature representation: Including the "hour" feature provides a more comprehensive representation of the dataset.<br>
It enriches the feature space and allows the model to learn more nuanced relationships between the input features and the target variable. This improved feature representation can enhance the model's capacity to capture the underlying patterns and make accurate predictions.

Overall, the addition of the "hour" feature in the bike sharing dataset led to a direct improvement in the Kaggle score. It allowed the model to leverage time-related patterns, user behavior, feature interactions, and improved feature representation, leading to more accurate predictions of bike sharing demand. This demonstrates the importance of feature engineering and the potential impact it can have on model performance and the discovery of valuable insights from the data.

The hyperparameter tuning performed on the individual models within AutoGluon resulted in a further improvement in the Kaggle score, which increased to 0.52822.<br> 
This improvement suggests that optimizing the hyperparameters of the models can have a significant impact on their performance and the overall predictive accuracy of the ensemble.

In this case, the hyperparameters that were tuned are specific to the individual models used in AutoGluon, namely the Gradient Boosting Machine (GBM) and Random Forest (RF). 
The hyperparameters set for each model were as follows:

* GBM: The hyperparameter tuned for the GBM model is "num_boost_round," which specifies the number of boosting rounds or iterations. By increasing the number of boosting rounds to 100, the GBM model can potentially learn more complex relationships and improve its predictive performance.

* RF: The hyperparameter tuned for the RF model is "n_estimators," which represents the number of decision trees in the random forest ensemble. By setting the number of estimators to 100, the RF model can benefit from a larger ensemble size, potentially capturing a wider range of patterns and improving its predictive accuracy.

By tuning these hyperparameters, the models were able to adapt and optimize their performance based on the characteristics of the bike sharing dataset, resulting in a better Kaggle score.

It's important to note that hyperparameter tuning is a crucial step in model optimization, as it allows the models to find the best configuration for their respective algorithms. The specific hyperparameters and their optimal values may vary depending on the dataset and the characteristics of the problem at hand.

