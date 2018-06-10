![](/images/logo_nrel.jpg)
# Predicting Public Transit Utilization

### Can a machine learning model predict whether a given trip will be serviced by public transit?
### If so, which features most strongly contribute to the prediction?

## Data:
The National Renewable Energy Laboratory Transportation Secure Data Center data can be found [here](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-cleansed-data.html)

These datasets consist of travel surveys distributed by several states and collected by NREL. For the preliminary research, I will use the California Household Travel Survey (CHTS) before scaling up to combine surveys from all states.

![](images/features.png)

## EDA:
There are thirty different transportation modes (plus two 'unknown' categories), of which fifteen indicate public transit. These were converted into binary categories with '1' indicating public transit.

![](images/modes.png)

In nearly 24% of the records, transportation mode is missing; as these records are not useful for modeling, they were dropped.

![](images/perc_missing.png)

### Imbalanced Classes (<4% transit)

![](images/meta-chart.png)

Over-/under-sample: SMOTEENN

## Preliminary Results:

![](images/preresults1.png)

![](images/preresults2.png)

## Improved Results:

AdaBoostClassifier(learning_rate=1, n_estimators=500)

![](images/post-results.png)

Most import features from AdaBoost:

![](images/ada_importances.png)

## Future Considerations:

1. Scale up to national datasets.
2. Incorporate NREL's secure latitude/longitude data.

##### Citation:

Transportation Secure Data Center." (2017). National Renewable Energy Laboratory. www.nrel.gov/tsdc.
