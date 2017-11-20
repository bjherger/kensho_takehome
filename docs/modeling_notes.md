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
 - Zeror: Until 2:30
 - Pipeline for test eval: until 3:00
 - Model
 - Writeup: 4:15
 - End: 5:00