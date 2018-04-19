![](/src/logo_nrel.jpg)
# Predicting Public Transportation

### Can a machine-learning model predict whether a given trip will be serviced by public transit?
### If so, which features most strongly contribute to the prediction?

## Data:
The National Renewable Energy Laboratory Transportation Secure Data Center data can be found [here](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-cleansed-data.html)

These datasets consist of travel surveys distributed by several states and collected by NREL. For the preliminary research, I will use the California Household Travel Survey (CHTS) before scaling up to combine all of the state surveys.

## EDA:
There are thirty different transportation modes (plus two 'unknown' categories), of which fifteen indicate public transit. These were encoded into binary categories with '1' indicating public transit.

![](src/modes.png)

In nearly 24% of the records, transportation mode is missing; as these records are not useful for modeling, they were dropped.

![](src/perc_missing.png)

![](src/heatmap.png)

### Imbalanced Classes (~3% transit)

![](src/meta-chart.png)

Over-/under-sample: SMOTEENN

## Preliminary Results:

![](src/preresults1.png)
![](src/preresults2.png)


##### Citations:
https://github.com/scikit-learn-contrib/imbalanced-learn

Transportation Secure Data Center." (2017). National Renewable Energy Laboratory. www.nrel.gov/tsdc.
