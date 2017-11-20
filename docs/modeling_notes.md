# Modeling Notes

## 2017-11-20

### Instructions

Meta
 - Files stored online as `.tar.gz`
 - Files small enough to include w/ repo
 
Model

 - Model 0: ZeroR, always respond GRAND LARCENY
 - Model 1: Up to me

Data

 - Train / test data sets
 - Both datasets contain same schema (include response)
 
Response:

 - Approach, reason for approach
 - Key observations
 - Improvements given 1 hour, 1 week
 
### EDA

Performed using Tableau for geo analysis

 - Extracting lat long from location_1
 - Lat long well behaved
 - Vast majority of crime is Grand larceny
 
- Model ideas
  - Coarse filter: Grand larceny or not
  - Bin time of day into morning, day, evening, night
  - Weekday / weekend
  - Dummy out Burough
  - Manhattan or not filter

Tentative schedule
 - Start: 1:15
 - EDA: Until 215
 - ZeroR: Until 2:30
 - Pipeline for test eval: until 3:00
 - Model
 - Writeup: 4:15
 - End: 5:00
 
### ZeroR

(Guess the majority class)

Start: 2:07
End: 2:30

 - Checking if SKLearn has a ZeroR implementation
 - [DummyClassifier seems to be ZeroR like](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
 - Implemented dummy classifier w/ constant reponse. 
 - Dummy accuracy: .408355132927
 
### Creating keras model

Start: 2:30
End: 4:15

 - Using DL due to (mostly) numeric data, multiple output classes
 - Subsetting to regressors that are numerical in the input data set
 - Looking up `compstat_`
 - Compstat_ represents when the police report was created
 
Naturally numerical:

 - occurrence_day
 - occurrence_day
 - occurrence_year
 - compstat_month
 - compstat_day
 - compstat_year
 - lat
 - long

Easy to conver to numerical

 - occurence_month (mapping)
 - Borough (dummy or `MANHATTAN` OR NOT)
 - Jurisdiction (dummy or majority case or not)
 - occurence_date epoch time or something
 
 