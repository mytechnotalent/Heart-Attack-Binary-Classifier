# Heart Attack Binary Classificaton

### [dataset](https://www.kaggle.com/datasets/tarekmuhammed/patients-data-for-medical-field)

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: [Apache 2.0](https://apache.org/licenses/LICENSE-2.0)

## **Executive Summary**

### **Objective**

This report presents the development and evaluation of a binary classification model designed to predict heart attack occurrences within a national healthcare dataset. Leveraging data on demographics, health conditions, and preventive measures, the model aims to enhance insights into population health, highlighting key risk factors associated with heart attacks.

### **Data Insights**

**1. Continuous Variables Analysis**
- **Height and Weight Distributions:** HeightInMeters approximates a normal distribution, while WeightInKilograms skews slightly towards higher values. BMI exhibits a right-skew, indicating a predominance of healthy to overweight individuals, with fewer in the higher BMI ranges.
- **Prevalence of Health Conditions:** Binary health conditions, such as HadAngina and HadStroke, are generally rare, with most individuals unaffected. However, conditions like HadArthritis, AlcoholDrinkers, and ChestScan show higher prevalence, indicating they are more commonly observed.
- **Vaccine and Preventive Measures:** Variables like FluVaxLast12, PneumoVaxEver, HighRiskLastYear, and CovidPos suggest that most individuals have not experienced these events, with a small proportion indicating positive cases.

**2. Categorical Variables Analysis**
- **Geographic Distribution:** The “Count of State” chart shows a wide geographic distribution of respondents, with higher counts in populous states like California and Texas.
- **Gender Representation:** The distribution between “Male” and “Female” is nearly balanced.
- **General Health Ratings:** Most respondents rate their health as “Very Good” or “Good,” with fewer reporting “Poor” health, suggesting a largely healthy population.
- **Age Distribution:** Middle-aged and senior groups are well-represented, reflecting a slight skew towards older age categories.
- **Diabetes and Smoking Status:** A majority report not having diabetes, though a notable segment does. Smoking data shows that most respondents have “Never smoked,” with e-cigarette usage relatively low.
- **Race and Preventive Care:** “White only, Non-Hispanic” is the predominant racial group. Tetanus vaccination data suggests some gaps in preventive care awareness, as many respondents are unsure about their shot type.

**3. Advanced Visualizations**
- **Demographic and Health Patterns:** State-level data shows variation in respondent counts, with larger numbers in states like Washington and Texas. Health ratings skew towards “Very Good” or “Good.” Age distribution favors older groups, particularly those aged 60-79, with diabetes prevalence increasing in these age ranges.
- **Smoking and Lifestyle Factors:** A majority report “Never smoked,” with low e-cigarette usage. Tetanus vaccination records are mixed, suggesting a need for improved awareness in preventive care. Racial distribution is mostly “White, Non-Hispanic,” with smaller segments representing other groups.
- **Self-Reported Health and Health Conditions:** Conditions like asthma, COPD, and depressive disorders are associated with poorer self-rated health, as are functional limitations (e.g., difficulty walking, dressing).
  
### **Model Development and Selection**

**Model Evaluation**
- **Best Model:** Achieved an accuracy of 94.74% and an ROC AUC of 0.8861 with 111 coefficients, indicating strong performance and predictive robustness. Key variables include:
  - **Geographic Influence:** States such as South Dakota (0.5115, p = 0.0001) and Maine (0.4423, p = 0.0004) show higher probabilities of positive outcomes.
  - **Age Factor:** Age is a strong predictor, with the oldest groups (80+ with a coefficient of 2.4317, p < 0.0001) having the highest likelihoods.
  - **Health Conditions:** Variables like HadAngina (2.4228, p < 0.0001) and HadStroke (0.8598, p < 0.0001) show strong positive associations.
  - **Lifestyle and Preventive Measures:** Smoking status and flu vaccination last year show negative associations, while chest scans and pneumococcal vaccine uptake contribute positively.

### **Implications**

The best model demonstrates high predictive accuracy and effectively highlights associations between age, health conditions, and preventive measures with health outcomes. Its reliability and robust feature set make it suitable for population health monitoring and public health interventions.

### **Recommendations and Next Steps**

1. **Address Class Imbalance:** Implement resampling or algorithmic techniques to mitigate class imbalance.
2. **Feature Engineering:** Investigate potential interactions among demographic and health variables.
3. **External Validation:** Validate the model with external datasets for broader applicability.
4. **Public Health Deployment:** Deploy the model within health systems for monitoring and intervention support.

### **Conclusion**

The development of this model underscores the value of health data analytics in public health strategies. By addressing data challenges and refining feature selection, this model holds promise for enhancing resource allocation, preventive care efforts, and health outcome predictions across diverse populations.

## Import Main Modules & Dataset


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import itertools
```


```python
import statsmodels.formula.api as smf
```


```python
df = pd.read_excel('Patients Data ( Used for Heart Disease Prediction ).xlsx')
```

## Perform Basic Analysis


```python
df.shape
```




    (237630, 35)




```python
df.dtypes
```




    PatientID                      int64
    State                         object
    Sex                           object
    GeneralHealth                 object
    AgeCategory                   object
    HeightInMeters               float64
    WeightInKilograms            float64
    BMI                          float64
    HadHeartAttack                 int64
    HadAngina                      int64
    HadStroke                      int64
    HadAsthma                      int64
    HadSkinCancer                  int64
    HadCOPD                        int64
    HadDepressiveDisorder          int64
    HadKidneyDisease               int64
    HadArthritis                   int64
    HadDiabetes                   object
    DeafOrHardOfHearing            int64
    BlindOrVisionDifficulty        int64
    DifficultyConcentrating        int64
    DifficultyWalking              int64
    DifficultyDressingBathing      int64
    DifficultyErrands              int64
    SmokerStatus                  object
    ECigaretteUsage               object
    ChestScan                      int64
    RaceEthnicityCategory         object
    AlcoholDrinkers                int64
    HIVTesting                     int64
    FluVaxLast12                   int64
    PneumoVaxEver                  int64
    TetanusLast10Tdap             object
    HighRiskLastYear               int64
    CovidPos                       int64
    dtype: object




```python
_ = [print(f'{df[column].value_counts()}\n') for column in df.columns]
```

    PatientID
    1         1
    158425    1
    158413    1
    158414    1
    158415    1
             ..
    79215     1
    79216     1
    79217     1
    79218     1
    237630    1
    Name: count, Length: 237630, dtype: int64
    
    State
    Washington              14241
    Maryland                 8817
    Minnesota                8712
    Ohio                     8700
    New York                 8625
    Texas                    7267
    Florida                  7124
    Kansas                   6000
    Wisconsin                5890
    Maine                    5709
    Iowa                     5492
    Indiana                  5393
    South Carolina           5360
    Virginia                 5358
    Arizona                  5302
    Hawaii                   5262
    Utah                     5212
    Michigan                 5206
    Massachusetts            5164
    Nebraska                 5008
    Colorado                 4973
    Georgia                  4860
    California               4801
    Connecticut              4765
    Vermont                  4569
    South Dakota             4280
    Montana                  4155
    Missouri                 4042
    New Jersey               3833
    New Hampshire            3564
    Puerto Rico              3550
    Idaho                    3394
    Alaska                   3100
    Rhode Island             3006
    Louisiana                2958
    Oregon                   2907
    Oklahoma                 2899
    West Virginia            2898
    New Mexico               2894
    Arkansas                 2881
    Tennessee                2659
    Pennsylvania             2623
    North Carolina           2531
    Illinois                 2516
    North Dakota             2435
    Mississippi              2416
    Kentucky                 2381
    Wyoming                  2345
    Delaware                 2079
    Alabama                  1880
    Nevada                   1693
    District of Columbia     1648
    Guam                     1521
    Virgin Islands            732
    Name: count, dtype: int64
    
    Sex
    Female    123293
    Male      114337
    Name: count, dtype: int64
    
    GeneralHealth
    Very good    83520
    Good         74950
    Excellent    39911
    Fair         29965
    Poor          9284
    Name: count, dtype: int64
    
    AgeCategory
    Age 65 to 69       27547
    Age 60 to 64       25685
    Age 70 to 74       24946
    Age 55 to 59       21422
    Age 50 to 54       19154
    Age 75 to 79       17679
    Age 80 or older    17544
    Age 40 to 44       16228
    Age 45 to 49       16095
    Age 35 to 39       14982
    Age 30 to 34       12825
    Age 18 to 24       12777
    Age 25 to 29       10746
    Name: count, dtype: int64
    
    HeightInMeters
    1.68    20909
    1.63    20104
    1.70    19356
    1.78    18828
    1.65    18197
            ...  
    1.92        1
    1.44        1
    1.18        1
    1.09        1
    2.21        1
    Name: count, Length: 101, dtype: int64
    
    WeightInKilograms
    90.720001     12677
    81.650002     11603
    68.040001     10026
    72.570000      9987
    77.110001      9370
                  ...  
    166.470001        1
    205.020004        1
    178.259995        1
    205.479996        1
    30.389999         1
    Name: count, Length: 513, dtype: int64
    
    BMI
    26.629999    2612
    27.459999    1965
    27.440001    1896
    24.410000    1861
    27.120001    1831
                 ... 
    27.420000       1
    20.570000       1
    29.219999       1
    52.340000       1
    45.279999       1
    Name: count, Length: 3503, dtype: int64
    
    HadHeartAttack
    0    224429
    1     13201
    Name: count, dtype: int64
    
    HadAngina
    0    223013
    1     14617
    Name: count, dtype: int64
    
    HadStroke
    0    227702
    1      9928
    Name: count, dtype: int64
    
    HadAsthma
    0    202338
    1     35292
    Name: count, dtype: int64
    
    HadSkinCancer
    0    217378
    1     20252
    Name: count, dtype: int64
    
    HadCOPD
    0    219028
    1     18602
    Name: count, dtype: int64
    
    HadDepressiveDisorder
    0    188734
    1     48896
    Name: count, dtype: int64
    
    HadKidneyDisease
    0    226601
    1     11029
    Name: count, dtype: int64
    
    HadArthritis
    0    155281
    1     82349
    Name: count, dtype: int64
    
    HadDiabetes
    No                                         197463
    Yes                                         33055
    No, pre-diabetes or borderline diabetes      5211
    Yes, but only during pregnancy (female)      1901
    Name: count, dtype: int64
    
    DeafOrHardOfHearing
    0    217090
    1     20540
    Name: count, dtype: int64
    
    BlindOrVisionDifficulty
    0    225643
    1     11987
    Name: count, dtype: int64
    
    DifficultyConcentrating
    0    212129
    1     25501
    Name: count, dtype: int64
    
    DifficultyWalking
    0    202239
    1     35391
    Name: count, dtype: int64
    
    DifficultyDressingBathing
    0    229426
    1      8204
    Name: count, dtype: int64
    
    DifficultyErrands
    0    221574
    1     16056
    Name: count, dtype: int64
    
    SmokerStatus
    Never smoked                             142390
    Former smoker                             66193
    Current smoker - now smokes every day     21148
    Current smoker - now smokes some days      7899
    Name: count, dtype: int64
    
    ECigaretteUsage
    Never used e-cigarettes in my entire life    183446
    Not at all (right now)                        41963
    Use them some days                             6468
    Use them every day                             5753
    Name: count, dtype: int64
    
    ChestScan
    0    136176
    1    101454
    Name: count, dtype: int64
    
    RaceEthnicityCategory
    White only, Non-Hispanic         179369
    Hispanic                          22023
    Black only, Non-Hispanic          19053
    Other race only, Non-Hispanic     11802
    Multiracial, Non-Hispanic          5383
    Name: count, dtype: int64
    
    AlcoholDrinkers
    1    129576
    0    108054
    Name: count, dtype: int64
    
    HIVTesting
    0    156195
    1     81435
    Name: count, dtype: int64
    
    FluVaxLast12
    1    126397
    0    111233
    Name: count, dtype: int64
    
    PneumoVaxEver
    0    140885
    1     96745
    Name: count, dtype: int64
    
    TetanusLast10Tdap
    No, did not receive any tetanus shot in the past 10 years    79370
    Yes, received tetanus shot but not sure what type            71538
    Yes, received Tdap                                           67418
    Yes, received tetanus shot, but not Tdap                     19304
    Name: count, dtype: int64
    
    HighRiskLastYear
    0    227454
    1     10176
    Name: count, dtype: int64
    
    CovidPos
    0    167306
    1     70324
    Name: count, dtype: int64
    



```python
df.nunique()
```




    PatientID                    237630
    State                            54
    Sex                               2
    GeneralHealth                     5
    AgeCategory                      13
    HeightInMeters                  101
    WeightInKilograms               513
    BMI                            3503
    HadHeartAttack                    2
    HadAngina                         2
    HadStroke                         2
    HadAsthma                         2
    HadSkinCancer                     2
    HadCOPD                           2
    HadDepressiveDisorder             2
    HadKidneyDisease                  2
    HadArthritis                      2
    HadDiabetes                       4
    DeafOrHardOfHearing               2
    BlindOrVisionDifficulty           2
    DifficultyConcentrating           2
    DifficultyWalking                 2
    DifficultyDressingBathing         2
    DifficultyErrands                 2
    SmokerStatus                      4
    ECigaretteUsage                   4
    ChestScan                         2
    RaceEthnicityCategory             5
    AlcoholDrinkers                   2
    HIVTesting                        2
    FluVaxLast12                      2
    PneumoVaxEver                     2
    TetanusLast10Tdap                 4
    HighRiskLastYear                  2
    CovidPos                          2
    dtype: int64



## Drop Unused Variables


```python
df.drop(['PatientID'], 
        axis=1, 
        inplace=True)
```

## Verify/Handle Missing Values 


```python
df.isna().sum()
```




    State                        0
    Sex                          0
    GeneralHealth                0
    AgeCategory                  0
    HeightInMeters               0
    WeightInKilograms            0
    BMI                          0
    HadHeartAttack               0
    HadAngina                    0
    HadStroke                    0
    HadAsthma                    0
    HadSkinCancer                0
    HadCOPD                      0
    HadDepressiveDisorder        0
    HadKidneyDisease             0
    HadArthritis                 0
    HadDiabetes                  0
    DeafOrHardOfHearing          0
    BlindOrVisionDifficulty      0
    DifficultyConcentrating      0
    DifficultyWalking            0
    DifficultyDressingBathing    0
    DifficultyErrands            0
    SmokerStatus                 0
    ECigaretteUsage              0
    ChestScan                    0
    RaceEthnicityCategory        0
    AlcoholDrinkers              0
    HIVTesting                   0
    FluVaxLast12                 0
    PneumoVaxEver                0
    TetanusLast10Tdap            0
    HighRiskLastYear             0
    CovidPos                     0
    dtype: int64



## Create Input & Output Vars


```python
cat_input_vars = ['State',
                  'Sex',
                  'GeneralHealth',
                  'AgeCategory',
                  'HadDiabetes',
                  'SmokerStatus',
                  'ECigaretteUsage',
                  'RaceEthnicityCategory',
                  'TetanusLast10Tdap']
```


```python
cont_input_vars = ['HeightInMeters',
                   'WeightInKilograms',
                   'BMI',
                   'HadAngina',
                   'HadStroke',
                   'HadAsthma',
                   'HadSkinCancer',
                   'HadCOPD',
                   'HadDepressiveDisorder',
                   'HadKidneyDisease',
                   'HadArthritis',
                   'DeafOrHardOfHearing',
                   'BlindOrVisionDifficulty',
                   'DifficultyConcentrating',
                   'DifficultyWalking',
                   'DifficultyDressingBathing',
                   'DifficultyErrands',
                   'ChestScan',
                   'AlcoholDrinkers',
                   'HIVTesting',
                   'FluVaxLast12',
                   'PneumoVaxEver',
                   'HighRiskLastYear',
                   'CovidPos']
```


```python
target= 'HadHeartAttack'
```

## Exploratory Data Analysis

### Marginal Distributions Continuous Variables: Histograms and Density Plots
* The dataset reveals expected patterns in both continuous and binary variables. Continuous variables like HeightInMeters and WeightInKilograms follow typical human distributions, with Height being nearly normal and Weight slightly skewed towards higher values; BMI shows a right skew, indicating a prevalence of healthy to overweight individuals with fewer in higher BMI ranges. Most binary health conditions, including HadAngina, HadStroke, and others, are rare, with the majority of individuals unaffected, suggesting a generally healthy population with occasional cases. However, conditions like HadArthritis, AlcoholDrinkers, and ChestScan appear more evenly distributed, reflecting higher prevalence. Similarly, indicators such as FluVaxLast12, PneumoVaxEver, HighRiskLastYear, and CovidPos show that most individuals did not experience these events, though a small proportion did, indicating occasional positive cases within the population. Overall, the data suggests a generally healthy population, with specific health issues and vaccine uptake showing variability.


```python
for cont in cont_input_vars:
    sns.displot(data=df,
                x=cont,
                kind='hist',
                bins=20,
                kde=True,
                aspect=2.0,
                height=6)
    plt.xticks(rotation=90)
    plt.title(f'Marginal Distribution of {cont}')
    plt.show()
```


    
![png](README_files/README_22_0.png)
    



    
![png](README_files/README_22_1.png)
    



    
![png](README_files/README_22_2.png)
    



    
![png](README_files/README_22_3.png)
    



    
![png](README_files/README_22_4.png)
    



    
![png](README_files/README_22_5.png)
    



    
![png](README_files/README_22_6.png)
    



    
![png](README_files/README_22_7.png)
    



    
![png](README_files/README_22_8.png)
    



    
![png](README_files/README_22_9.png)
    



    
![png](README_files/README_22_10.png)
    



    
![png](README_files/README_22_11.png)
    



    
![png](README_files/README_22_12.png)
    



    
![png](README_files/README_22_13.png)
    



    
![png](README_files/README_22_14.png)
    



    
![png](README_files/README_22_15.png)
    



    
![png](README_files/README_22_16.png)
    



    
![png](README_files/README_22_17.png)
    



    
![png](README_files/README_22_18.png)
    



    
![png](README_files/README_22_19.png)
    



    
![png](README_files/README_22_20.png)
    



    
![png](README_files/README_22_21.png)
    



    
![png](README_files/README_22_22.png)
    



    
![png](README_files/README_22_23.png)
    


### Marginal Distributions Categorical Variables: Bar Charts
* The categorical data visualizations reveal various demographic and health-related insights. The “Count of State” chart indicates a wide geographic distribution of respondents, with higher counts in states like California and Texas. Gender distribution between “Male” and “Female” is nearly equal, showing balanced representation. General health ratings lean towards “Very Good” and “Good,” suggesting a largely healthy population, with fewer individuals reporting “Poor” health. Age categories are well-distributed, with higher counts in middle-aged and senior groups. In the “HadDiabetes” category, the majority have not been diagnosed, though there is a notable segment with diabetes. Smoking status shows a high proportion of “Never smoked” individuals, followed by “Former smokers,” while e-cigarette usage remains relatively low. The “RaceEthnicityCategory” chart is predominantly “White only, Non-Hispanic,” with smaller representations of other racial groups. The “TetanusLast10Tdap” data reveals that many respondents either did not receive a recent shot or are unsure about the type received, highlighting potential gaps in preventive care awareness.


```python
for cat in cat_input_vars:
    sns.catplot(
        data=df,
        x=cat,
        kind='count',
        height=6,
        aspect=2.0,
        legend=False
    )
    plt.xticks(rotation=90)
    plt.title(f'Count of {cat}')
    plt.show()
```


    
![png](README_files/README_24_0.png)
    



    
![png](README_files/README_24_1.png)
    



    
![png](README_files/README_24_2.png)
    



    
![png](README_files/README_24_3.png)
    



    
![png](README_files/README_24_4.png)
    



    
![png](README_files/README_24_5.png)
    



    
![png](README_files/README_24_6.png)
    



    
![png](README_files/README_24_7.png)
    



    
![png](README_files/README_24_8.png)
    


### Categorical-to-Categorical Relationships or Combinations: Dodged Bar Charts and Heatmaps
* The visualizations provide a comprehensive analysis of demographic distributions, health conditions, and lifestyle factors across multiple categories, revealing significant geographic, gender, age, racial, and health-related patterns. State-level data shows variation in respondent counts, with states like Washington and Texas having higher numbers. Gender distribution is balanced, and general health ratings are mostly “Very Good” or “Good,” with fewer respondents reporting “Poor” health. Age distribution skews towards older adults, particularly those aged 60-79, and diabetes prevalence increases with age, especially in older groups. Smoking data indicates a majority of respondents have “Never smoked,” while e-cigarette usage remains low. Racially, “White, Non-Hispanic” is the most represented group, with smaller segments from other racial and ethnic backgrounds. Tetanus vaccination records vary, showing a mix of Tdap recipients, other shot types, or no recent shots, suggesting potential gaps in preventive care awareness. Overall, the data points to a generally healthy population with diverse lifestyle habits and health awareness, alongside observed disparities across demographic segments, particularly in health status, diabetes prevalence, and preventive care measures.


```python
pairwise_comparisons = list(itertools.combinations(cat_input_vars, 2))
for cat1, cat2 in pairwise_comparisons:
    sns.catplot(data=df,
                x=cat1, 
                hue=cat2, 
                kind='count', 
                aspect=2, 
                height=6)
    plt.title(f'{cat1} vs. {cat2}')
    plt.xticks(rotation=90)
    plt.show()
```


    
![png](README_files/README_26_0.png)
    



    
![png](README_files/README_26_1.png)
    



    
![png](README_files/README_26_2.png)
    



    
![png](README_files/README_26_3.png)
    



    
![png](README_files/README_26_4.png)
    



    
![png](README_files/README_26_5.png)
    



    
![png](README_files/README_26_6.png)
    



    
![png](README_files/README_26_7.png)
    



    
![png](README_files/README_26_8.png)
    



    
![png](README_files/README_26_9.png)
    



    
![png](README_files/README_26_10.png)
    



    
![png](README_files/README_26_11.png)
    



    
![png](README_files/README_26_12.png)
    



    
![png](README_files/README_26_13.png)
    



    
![png](README_files/README_26_14.png)
    



    
![png](README_files/README_26_15.png)
    



    
![png](README_files/README_26_16.png)
    



    
![png](README_files/README_26_17.png)
    



    
![png](README_files/README_26_18.png)
    



    
![png](README_files/README_26_19.png)
    



    
![png](README_files/README_26_20.png)
    



    
![png](README_files/README_26_21.png)
    



    
![png](README_files/README_26_22.png)
    



    
![png](README_files/README_26_23.png)
    



    
![png](README_files/README_26_24.png)
    



    
![png](README_files/README_26_25.png)
    



    
![png](README_files/README_26_26.png)
    



    
![png](README_files/README_26_27.png)
    



    
![png](README_files/README_26_28.png)
    



    
![png](README_files/README_26_29.png)
    



    
![png](README_files/README_26_30.png)
    



    
![png](README_files/README_26_31.png)
    



    
![png](README_files/README_26_32.png)
    



    
![png](README_files/README_26_33.png)
    



    
![png](README_files/README_26_34.png)
    



    
![png](README_files/README_26_35.png)
    



```python
pairwise_comparisons = list(itertools.combinations(cat_input_vars, 2))
for cat1, cat2 in pairwise_comparisons:
    crosstab = pd.crosstab(df[cat1], df[cat2])
    plt.figure(figsize=(12, 10))
    sns.heatmap(crosstab, 
                annot=True, 
                annot_kws={'size': 10}, 
                fmt='d', 
                cbar=False)
    plt.title(f'{cat1} vs. {cat2}')
    plt.xticks(rotation=90)
    plt.show()
```


    
![png](README_files/README_27_0.png)
    



    
![png](README_files/README_27_1.png)
    



    
![png](README_files/README_27_2.png)
    



    
![png](README_files/README_27_3.png)
    



    
![png](README_files/README_27_4.png)
    



    
![png](README_files/README_27_5.png)
    



    
![png](README_files/README_27_6.png)
    



    
![png](README_files/README_27_7.png)
    



    
![png](README_files/README_27_8.png)
    



    
![png](README_files/README_27_9.png)
    



    
![png](README_files/README_27_10.png)
    



    
![png](README_files/README_27_11.png)
    



    
![png](README_files/README_27_12.png)
    



    
![png](README_files/README_27_13.png)
    



    
![png](README_files/README_27_14.png)
    



    
![png](README_files/README_27_15.png)
    



    
![png](README_files/README_27_16.png)
    



    
![png](README_files/README_27_17.png)
    



    
![png](README_files/README_27_18.png)
    



    
![png](README_files/README_27_19.png)
    



    
![png](README_files/README_27_20.png)
    



    
![png](README_files/README_27_21.png)
    



    
![png](README_files/README_27_22.png)
    



    
![png](README_files/README_27_23.png)
    



    
![png](README_files/README_27_24.png)
    



    
![png](README_files/README_27_25.png)
    



    
![png](README_files/README_27_26.png)
    



    
![png](README_files/README_27_27.png)
    



    
![png](README_files/README_27_28.png)
    



    
![png](README_files/README_27_29.png)
    



    
![png](README_files/README_27_30.png)
    



    
![png](README_files/README_27_31.png)
    



    
![png](README_files/README_27_32.png)
    



    
![png](README_files/README_27_33.png)
    



    
![png](README_files/README_27_34.png)
    



    
![png](README_files/README_27_35.png)
    


### Categorical-to-Continuous Relationships or Conditional Distributions: Box Plots, Violin Plots and Point Plots
* The visualizations illustrate relationships between health conditions, demographic attributes, and self-reported health across U.S. states. Depressive disorder and COVID-19 positivity rates show consistent prevalence across states, while depressive disorders are notably more frequent among females and younger adults (ages 25-39) than in older age groups. Health conditions such as asthma, COPD, and depression are strongly associated with poor self-rated health, as are functional difficulties (concentration, walking, dressing, and errands). In contrast, lifestyle factors like chest scans and alcohol consumption are evenly distributed across health categories, indicating no strong association with self-rated health. The data also reveal state-level variations in preventive care measures (flu and tetanus vaccinations), with differences across racial and ethnic groups, highlighting disparities in high-risk behaviors and health outcomes. This comprehensive overview emphasizes complex associations between demographic factors, preventive care, and health outcomes, pinpointing gaps and areas for improvement in health awareness and care access across diverse U.S. populations.


```python
for cat_input_var in cat_input_vars:
    for cont_input_var in cont_input_vars:
        sns.catplot(data=df,
                    x=cat_input_var, 
                    y=cont_input_var, 
                    kind='box', 
                    aspect=2.5, 
                    height=5)
        plt.title(f'{cont_input_var} vs. {cat_input_var}')
        plt.xticks(rotation=90)
        plt.show()
```


    
![png](README_files/README_29_0.png)
    



    
![png](README_files/README_29_1.png)
    



    
![png](README_files/README_29_2.png)
    



    
![png](README_files/README_29_3.png)
    



    
![png](README_files/README_29_4.png)
    



    
![png](README_files/README_29_5.png)
    



    
![png](README_files/README_29_6.png)
    



    
![png](README_files/README_29_7.png)
    



    
![png](README_files/README_29_8.png)
    



    
![png](README_files/README_29_9.png)
    



    
![png](README_files/README_29_10.png)
    



    
![png](README_files/README_29_11.png)
    



    
![png](README_files/README_29_12.png)
    



    
![png](README_files/README_29_13.png)
    



    
![png](README_files/README_29_14.png)
    



    
![png](README_files/README_29_15.png)
    



    
![png](README_files/README_29_16.png)
    



    
![png](README_files/README_29_17.png)
    



    
![png](README_files/README_29_18.png)
    



    
![png](README_files/README_29_19.png)
    



    
![png](README_files/README_29_20.png)
    



    
![png](README_files/README_29_21.png)
    



    
![png](README_files/README_29_22.png)
    



    
![png](README_files/README_29_23.png)
    



    
![png](README_files/README_29_24.png)
    



    
![png](README_files/README_29_25.png)
    



    
![png](README_files/README_29_26.png)
    



    
![png](README_files/README_29_27.png)
    



    
![png](README_files/README_29_28.png)
    



    
![png](README_files/README_29_29.png)
    



    
![png](README_files/README_29_30.png)
    



    
![png](README_files/README_29_31.png)
    



    
![png](README_files/README_29_32.png)
    



    
![png](README_files/README_29_33.png)
    



    
![png](README_files/README_29_34.png)
    



    
![png](README_files/README_29_35.png)
    



    
![png](README_files/README_29_36.png)
    



    
![png](README_files/README_29_37.png)
    



    
![png](README_files/README_29_38.png)
    



    
![png](README_files/README_29_39.png)
    



    
![png](README_files/README_29_40.png)
    



    
![png](README_files/README_29_41.png)
    



    
![png](README_files/README_29_42.png)
    



    
![png](README_files/README_29_43.png)
    



    
![png](README_files/README_29_44.png)
    



    
![png](README_files/README_29_45.png)
    



    
![png](README_files/README_29_46.png)
    



    
![png](README_files/README_29_47.png)
    



    
![png](README_files/README_29_48.png)
    



    
![png](README_files/README_29_49.png)
    



    
![png](README_files/README_29_50.png)
    



    
![png](README_files/README_29_51.png)
    



    
![png](README_files/README_29_52.png)
    



    
![png](README_files/README_29_53.png)
    



    
![png](README_files/README_29_54.png)
    



    
![png](README_files/README_29_55.png)
    



    
![png](README_files/README_29_56.png)
    



    
![png](README_files/README_29_57.png)
    



    
![png](README_files/README_29_58.png)
    



    
![png](README_files/README_29_59.png)
    



    
![png](README_files/README_29_60.png)
    



    
![png](README_files/README_29_61.png)
    



    
![png](README_files/README_29_62.png)
    



    
![png](README_files/README_29_63.png)
    



    
![png](README_files/README_29_64.png)
    



    
![png](README_files/README_29_65.png)
    



    
![png](README_files/README_29_66.png)
    



    
![png](README_files/README_29_67.png)
    



    
![png](README_files/README_29_68.png)
    



    
![png](README_files/README_29_69.png)
    



    
![png](README_files/README_29_70.png)
    



    
![png](README_files/README_29_71.png)
    



    
![png](README_files/README_29_72.png)
    



    
![png](README_files/README_29_73.png)
    



    
![png](README_files/README_29_74.png)
    



    
![png](README_files/README_29_75.png)
    



    
![png](README_files/README_29_76.png)
    



    
![png](README_files/README_29_77.png)
    



    
![png](README_files/README_29_78.png)
    



    
![png](README_files/README_29_79.png)
    



    
![png](README_files/README_29_80.png)
    



    
![png](README_files/README_29_81.png)
    



    
![png](README_files/README_29_82.png)
    



    
![png](README_files/README_29_83.png)
    



    
![png](README_files/README_29_84.png)
    



    
![png](README_files/README_29_85.png)
    



    
![png](README_files/README_29_86.png)
    



    
![png](README_files/README_29_87.png)
    



    
![png](README_files/README_29_88.png)
    



    
![png](README_files/README_29_89.png)
    



    
![png](README_files/README_29_90.png)
    



    
![png](README_files/README_29_91.png)
    



    
![png](README_files/README_29_92.png)
    



    
![png](README_files/README_29_93.png)
    



    
![png](README_files/README_29_94.png)
    



    
![png](README_files/README_29_95.png)
    



    
![png](README_files/README_29_96.png)
    



    
![png](README_files/README_29_97.png)
    



    
![png](README_files/README_29_98.png)
    



    
![png](README_files/README_29_99.png)
    



    
![png](README_files/README_29_100.png)
    



    
![png](README_files/README_29_101.png)
    



    
![png](README_files/README_29_102.png)
    



    
![png](README_files/README_29_103.png)
    



    
![png](README_files/README_29_104.png)
    



    
![png](README_files/README_29_105.png)
    



    
![png](README_files/README_29_106.png)
    



    
![png](README_files/README_29_107.png)
    



    
![png](README_files/README_29_108.png)
    



    
![png](README_files/README_29_109.png)
    



    
![png](README_files/README_29_110.png)
    



    
![png](README_files/README_29_111.png)
    



    
![png](README_files/README_29_112.png)
    



    
![png](README_files/README_29_113.png)
    



    
![png](README_files/README_29_114.png)
    



    
![png](README_files/README_29_115.png)
    



    
![png](README_files/README_29_116.png)
    



    
![png](README_files/README_29_117.png)
    



    
![png](README_files/README_29_118.png)
    



    
![png](README_files/README_29_119.png)
    



    
![png](README_files/README_29_120.png)
    



    
![png](README_files/README_29_121.png)
    



    
![png](README_files/README_29_122.png)
    



    
![png](README_files/README_29_123.png)
    



    
![png](README_files/README_29_124.png)
    



    
![png](README_files/README_29_125.png)
    



    
![png](README_files/README_29_126.png)
    



    
![png](README_files/README_29_127.png)
    



    
![png](README_files/README_29_128.png)
    



    
![png](README_files/README_29_129.png)
    



    
![png](README_files/README_29_130.png)
    



    
![png](README_files/README_29_131.png)
    



    
![png](README_files/README_29_132.png)
    



    
![png](README_files/README_29_133.png)
    



    
![png](README_files/README_29_134.png)
    



    
![png](README_files/README_29_135.png)
    



    
![png](README_files/README_29_136.png)
    



    
![png](README_files/README_29_137.png)
    



    
![png](README_files/README_29_138.png)
    



    
![png](README_files/README_29_139.png)
    



    
![png](README_files/README_29_140.png)
    



    
![png](README_files/README_29_141.png)
    



    
![png](README_files/README_29_142.png)
    



    
![png](README_files/README_29_143.png)
    



    
![png](README_files/README_29_144.png)
    



    
![png](README_files/README_29_145.png)
    



    
![png](README_files/README_29_146.png)
    



    
![png](README_files/README_29_147.png)
    



    
![png](README_files/README_29_148.png)
    



    
![png](README_files/README_29_149.png)
    



    
![png](README_files/README_29_150.png)
    



    
![png](README_files/README_29_151.png)
    



    
![png](README_files/README_29_152.png)
    



    
![png](README_files/README_29_153.png)
    



    
![png](README_files/README_29_154.png)
    



    
![png](README_files/README_29_155.png)
    



    
![png](README_files/README_29_156.png)
    



    
![png](README_files/README_29_157.png)
    



    
![png](README_files/README_29_158.png)
    



    
![png](README_files/README_29_159.png)
    



    
![png](README_files/README_29_160.png)
    



    
![png](README_files/README_29_161.png)
    



    
![png](README_files/README_29_162.png)
    



    
![png](README_files/README_29_163.png)
    



    
![png](README_files/README_29_164.png)
    



    
![png](README_files/README_29_165.png)
    



    
![png](README_files/README_29_166.png)
    



    
![png](README_files/README_29_167.png)
    



    
![png](README_files/README_29_168.png)
    



    
![png](README_files/README_29_169.png)
    



    
![png](README_files/README_29_170.png)
    



    
![png](README_files/README_29_171.png)
    



    
![png](README_files/README_29_172.png)
    



    
![png](README_files/README_29_173.png)
    



    
![png](README_files/README_29_174.png)
    



    
![png](README_files/README_29_175.png)
    



    
![png](README_files/README_29_176.png)
    



    
![png](README_files/README_29_177.png)
    



    
![png](README_files/README_29_178.png)
    



    
![png](README_files/README_29_179.png)
    



    
![png](README_files/README_29_180.png)
    



    
![png](README_files/README_29_181.png)
    



    
![png](README_files/README_29_182.png)
    



    
![png](README_files/README_29_183.png)
    



    
![png](README_files/README_29_184.png)
    



    
![png](README_files/README_29_185.png)
    



    
![png](README_files/README_29_186.png)
    



    
![png](README_files/README_29_187.png)
    



    
![png](README_files/README_29_188.png)
    



    
![png](README_files/README_29_189.png)
    



    
![png](README_files/README_29_190.png)
    



    
![png](README_files/README_29_191.png)
    



    
![png](README_files/README_29_192.png)
    



    
![png](README_files/README_29_193.png)
    



    
![png](README_files/README_29_194.png)
    



    
![png](README_files/README_29_195.png)
    



    
![png](README_files/README_29_196.png)
    



    
![png](README_files/README_29_197.png)
    



    
![png](README_files/README_29_198.png)
    



    
![png](README_files/README_29_199.png)
    



    
![png](README_files/README_29_200.png)
    



    
![png](README_files/README_29_201.png)
    



    
![png](README_files/README_29_202.png)
    



    
![png](README_files/README_29_203.png)
    



    
![png](README_files/README_29_204.png)
    



    
![png](README_files/README_29_205.png)
    



    
![png](README_files/README_29_206.png)
    



    
![png](README_files/README_29_207.png)
    



    
![png](README_files/README_29_208.png)
    



    
![png](README_files/README_29_209.png)
    



    
![png](README_files/README_29_210.png)
    



    
![png](README_files/README_29_211.png)
    



    
![png](README_files/README_29_212.png)
    



    
![png](README_files/README_29_213.png)
    



    
![png](README_files/README_29_214.png)
    



    
![png](README_files/README_29_215.png)
    



```python
for cat_input_var in cat_input_vars:
    for cont_input_var in cont_input_vars:
        sns.catplot(data=df,
                    x=cat_input_var, 
                    y=cont_input_var, 
                    kind='violin', 
                    aspect=2.5, 
                    height=5)
        plt.title(f'{cont_input_var} vs. {cat_input_var}')
        plt.xticks(rotation=90)
        plt.show()
```


    
![png](README_files/README_30_0.png)
    



    
![png](README_files/README_30_1.png)
    



    
![png](README_files/README_30_2.png)
    



    
![png](README_files/README_30_3.png)
    



    
![png](README_files/README_30_4.png)
    



    
![png](README_files/README_30_5.png)
    



    
![png](README_files/README_30_6.png)
    



    
![png](README_files/README_30_7.png)
    



    
![png](README_files/README_30_8.png)
    



    
![png](README_files/README_30_9.png)
    



    
![png](README_files/README_30_10.png)
    



    
![png](README_files/README_30_11.png)
    



    
![png](README_files/README_30_12.png)
    



    
![png](README_files/README_30_13.png)
    



    
![png](README_files/README_30_14.png)
    



    
![png](README_files/README_30_15.png)
    



    
![png](README_files/README_30_16.png)
    



    
![png](README_files/README_30_17.png)
    



    
![png](README_files/README_30_18.png)
    



    
![png](README_files/README_30_19.png)
    



    
![png](README_files/README_30_20.png)
    



    
![png](README_files/README_30_21.png)
    



    
![png](README_files/README_30_22.png)
    



    
![png](README_files/README_30_23.png)
    



    
![png](README_files/README_30_24.png)
    



    
![png](README_files/README_30_25.png)
    



    
![png](README_files/README_30_26.png)
    



    
![png](README_files/README_30_27.png)
    



    
![png](README_files/README_30_28.png)
    



    
![png](README_files/README_30_29.png)
    



    
![png](README_files/README_30_30.png)
    



    
![png](README_files/README_30_31.png)
    



    
![png](README_files/README_30_32.png)
    



    
![png](README_files/README_30_33.png)
    



    
![png](README_files/README_30_34.png)
    



    
![png](README_files/README_30_35.png)
    



    
![png](README_files/README_30_36.png)
    



    
![png](README_files/README_30_37.png)
    



    
![png](README_files/README_30_38.png)
    



    
![png](README_files/README_30_39.png)
    



    
![png](README_files/README_30_40.png)
    



    
![png](README_files/README_30_41.png)
    



    
![png](README_files/README_30_42.png)
    



    
![png](README_files/README_30_43.png)
    



    
![png](README_files/README_30_44.png)
    



    
![png](README_files/README_30_45.png)
    



    
![png](README_files/README_30_46.png)
    



    
![png](README_files/README_30_47.png)
    



    
![png](README_files/README_30_48.png)
    



    
![png](README_files/README_30_49.png)
    



    
![png](README_files/README_30_50.png)
    



    
![png](README_files/README_30_51.png)
    



    
![png](README_files/README_30_52.png)
    



    
![png](README_files/README_30_53.png)
    



    
![png](README_files/README_30_54.png)
    



    
![png](README_files/README_30_55.png)
    



    
![png](README_files/README_30_56.png)
    



    
![png](README_files/README_30_57.png)
    



    
![png](README_files/README_30_58.png)
    



    
![png](README_files/README_30_59.png)
    



    
![png](README_files/README_30_60.png)
    



    
![png](README_files/README_30_61.png)
    



    
![png](README_files/README_30_62.png)
    



    
![png](README_files/README_30_63.png)
    



    
![png](README_files/README_30_64.png)
    



    
![png](README_files/README_30_65.png)
    



    
![png](README_files/README_30_66.png)
    



    
![png](README_files/README_30_67.png)
    



    
![png](README_files/README_30_68.png)
    



    
![png](README_files/README_30_69.png)
    



    
![png](README_files/README_30_70.png)
    



    
![png](README_files/README_30_71.png)
    



    
![png](README_files/README_30_72.png)
    



    
![png](README_files/README_30_73.png)
    



    
![png](README_files/README_30_74.png)
    



    
![png](README_files/README_30_75.png)
    



    
![png](README_files/README_30_76.png)
    



    
![png](README_files/README_30_77.png)
    



    
![png](README_files/README_30_78.png)
    



    
![png](README_files/README_30_79.png)
    



    
![png](README_files/README_30_80.png)
    



    
![png](README_files/README_30_81.png)
    



    
![png](README_files/README_30_82.png)
    



    
![png](README_files/README_30_83.png)
    



    
![png](README_files/README_30_84.png)
    



    
![png](README_files/README_30_85.png)
    



    
![png](README_files/README_30_86.png)
    



    
![png](README_files/README_30_87.png)
    



    
![png](README_files/README_30_88.png)
    



    
![png](README_files/README_30_89.png)
    



    
![png](README_files/README_30_90.png)
    



    
![png](README_files/README_30_91.png)
    



    
![png](README_files/README_30_92.png)
    



    
![png](README_files/README_30_93.png)
    



    
![png](README_files/README_30_94.png)
    



    
![png](README_files/README_30_95.png)
    



    
![png](README_files/README_30_96.png)
    



    
![png](README_files/README_30_97.png)
    



    
![png](README_files/README_30_98.png)
    



    
![png](README_files/README_30_99.png)
    



    
![png](README_files/README_30_100.png)
    



    
![png](README_files/README_30_101.png)
    



    
![png](README_files/README_30_102.png)
    



    
![png](README_files/README_30_103.png)
    



    
![png](README_files/README_30_104.png)
    



    
![png](README_files/README_30_105.png)
    



    
![png](README_files/README_30_106.png)
    



    
![png](README_files/README_30_107.png)
    



    
![png](README_files/README_30_108.png)
    



    
![png](README_files/README_30_109.png)
    



    
![png](README_files/README_30_110.png)
    



    
![png](README_files/README_30_111.png)
    



    
![png](README_files/README_30_112.png)
    



    
![png](README_files/README_30_113.png)
    



    
![png](README_files/README_30_114.png)
    



    
![png](README_files/README_30_115.png)
    



    
![png](README_files/README_30_116.png)
    



    
![png](README_files/README_30_117.png)
    



    
![png](README_files/README_30_118.png)
    



    
![png](README_files/README_30_119.png)
    



    
![png](README_files/README_30_120.png)
    



    
![png](README_files/README_30_121.png)
    



    
![png](README_files/README_30_122.png)
    



    
![png](README_files/README_30_123.png)
    



    
![png](README_files/README_30_124.png)
    



    
![png](README_files/README_30_125.png)
    



    
![png](README_files/README_30_126.png)
    



    
![png](README_files/README_30_127.png)
    



    
![png](README_files/README_30_128.png)
    



    
![png](README_files/README_30_129.png)
    



    
![png](README_files/README_30_130.png)
    



    
![png](README_files/README_30_131.png)
    



    
![png](README_files/README_30_132.png)
    



    
![png](README_files/README_30_133.png)
    



    
![png](README_files/README_30_134.png)
    



    
![png](README_files/README_30_135.png)
    



    
![png](README_files/README_30_136.png)
    



    
![png](README_files/README_30_137.png)
    



    
![png](README_files/README_30_138.png)
    



    
![png](README_files/README_30_139.png)
    



    
![png](README_files/README_30_140.png)
    



    
![png](README_files/README_30_141.png)
    



    
![png](README_files/README_30_142.png)
    



    
![png](README_files/README_30_143.png)
    



    
![png](README_files/README_30_144.png)
    



    
![png](README_files/README_30_145.png)
    



    
![png](README_files/README_30_146.png)
    



    
![png](README_files/README_30_147.png)
    



    
![png](README_files/README_30_148.png)
    



    
![png](README_files/README_30_149.png)
    



    
![png](README_files/README_30_150.png)
    



    
![png](README_files/README_30_151.png)
    



    
![png](README_files/README_30_152.png)
    



    
![png](README_files/README_30_153.png)
    



    
![png](README_files/README_30_154.png)
    



    
![png](README_files/README_30_155.png)
    



    
![png](README_files/README_30_156.png)
    



    
![png](README_files/README_30_157.png)
    



    
![png](README_files/README_30_158.png)
    



    
![png](README_files/README_30_159.png)
    



    
![png](README_files/README_30_160.png)
    



    
![png](README_files/README_30_161.png)
    



    
![png](README_files/README_30_162.png)
    



    
![png](README_files/README_30_163.png)
    



    
![png](README_files/README_30_164.png)
    



    
![png](README_files/README_30_165.png)
    



    
![png](README_files/README_30_166.png)
    



    
![png](README_files/README_30_167.png)
    



    
![png](README_files/README_30_168.png)
    



    
![png](README_files/README_30_169.png)
    



    
![png](README_files/README_30_170.png)
    



    
![png](README_files/README_30_171.png)
    



    
![png](README_files/README_30_172.png)
    



    
![png](README_files/README_30_173.png)
    



    
![png](README_files/README_30_174.png)
    



    
![png](README_files/README_30_175.png)
    



    
![png](README_files/README_30_176.png)
    



    
![png](README_files/README_30_177.png)
    



    
![png](README_files/README_30_178.png)
    



    
![png](README_files/README_30_179.png)
    



    
![png](README_files/README_30_180.png)
    



    
![png](README_files/README_30_181.png)
    



    
![png](README_files/README_30_182.png)
    



    
![png](README_files/README_30_183.png)
    



    
![png](README_files/README_30_184.png)
    



    
![png](README_files/README_30_185.png)
    



    
![png](README_files/README_30_186.png)
    



    
![png](README_files/README_30_187.png)
    



    
![png](README_files/README_30_188.png)
    



    
![png](README_files/README_30_189.png)
    



    
![png](README_files/README_30_190.png)
    



    
![png](README_files/README_30_191.png)
    



    
![png](README_files/README_30_192.png)
    



    
![png](README_files/README_30_193.png)
    



    
![png](README_files/README_30_194.png)
    



    
![png](README_files/README_30_195.png)
    



    
![png](README_files/README_30_196.png)
    



    
![png](README_files/README_30_197.png)
    



    
![png](README_files/README_30_198.png)
    



    
![png](README_files/README_30_199.png)
    



    
![png](README_files/README_30_200.png)
    



    
![png](README_files/README_30_201.png)
    



    
![png](README_files/README_30_202.png)
    



    
![png](README_files/README_30_203.png)
    



    
![png](README_files/README_30_204.png)
    



    
![png](README_files/README_30_205.png)
    



    
![png](README_files/README_30_206.png)
    



    
![png](README_files/README_30_207.png)
    



    
![png](README_files/README_30_208.png)
    



    
![png](README_files/README_30_209.png)
    



    
![png](README_files/README_30_210.png)
    



    
![png](README_files/README_30_211.png)
    



    
![png](README_files/README_30_212.png)
    



    
![png](README_files/README_30_213.png)
    



    
![png](README_files/README_30_214.png)
    



    
![png](README_files/README_30_215.png)
    



```python
for cat_input_var in cat_input_vars:
    for cont_input_var in cont_input_vars:
        sns.catplot(data=df,
                    x=cat_input_var, 
                    y=cont_input_var, 
                    kind='point', 
                    linestyles='',
                    aspect=2.5, 
                    height=5)
        plt.title(f'{cont_input_var} vs. {cat_input_var}')
        plt.xticks(rotation=90)
        plt.show()
```


    
![png](README_files/README_31_0.png)
    



    
![png](README_files/README_31_1.png)
    



    
![png](README_files/README_31_2.png)
    



    
![png](README_files/README_31_3.png)
    



    
![png](README_files/README_31_4.png)
    



    
![png](README_files/README_31_5.png)
    



    
![png](README_files/README_31_6.png)
    



    
![png](README_files/README_31_7.png)
    



    
![png](README_files/README_31_8.png)
    



    
![png](README_files/README_31_9.png)
    



    
![png](README_files/README_31_10.png)
    



    
![png](README_files/README_31_11.png)
    



    
![png](README_files/README_31_12.png)
    



    
![png](README_files/README_31_13.png)
    



    
![png](README_files/README_31_14.png)
    



    
![png](README_files/README_31_15.png)
    



    
![png](README_files/README_31_16.png)
    



    
![png](README_files/README_31_17.png)
    



    
![png](README_files/README_31_18.png)
    



    
![png](README_files/README_31_19.png)
    



    
![png](README_files/README_31_20.png)
    



    
![png](README_files/README_31_21.png)
    



    
![png](README_files/README_31_22.png)
    



    
![png](README_files/README_31_23.png)
    



    
![png](README_files/README_31_24.png)
    



    
![png](README_files/README_31_25.png)
    



    
![png](README_files/README_31_26.png)
    



    
![png](README_files/README_31_27.png)
    



    
![png](README_files/README_31_28.png)
    



    
![png](README_files/README_31_29.png)
    



    
![png](README_files/README_31_30.png)
    



    
![png](README_files/README_31_31.png)
    



    
![png](README_files/README_31_32.png)
    



    
![png](README_files/README_31_33.png)
    



    
![png](README_files/README_31_34.png)
    



    
![png](README_files/README_31_35.png)
    



    
![png](README_files/README_31_36.png)
    



    
![png](README_files/README_31_37.png)
    



    
![png](README_files/README_31_38.png)
    



    
![png](README_files/README_31_39.png)
    



    
![png](README_files/README_31_40.png)
    



    
![png](README_files/README_31_41.png)
    



    
![png](README_files/README_31_42.png)
    



    
![png](README_files/README_31_43.png)
    



    
![png](README_files/README_31_44.png)
    



    
![png](README_files/README_31_45.png)
    



    
![png](README_files/README_31_46.png)
    



    
![png](README_files/README_31_47.png)
    



    
![png](README_files/README_31_48.png)
    



    
![png](README_files/README_31_49.png)
    



    
![png](README_files/README_31_50.png)
    



    
![png](README_files/README_31_51.png)
    



    
![png](README_files/README_31_52.png)
    



    
![png](README_files/README_31_53.png)
    



    
![png](README_files/README_31_54.png)
    



    
![png](README_files/README_31_55.png)
    



    
![png](README_files/README_31_56.png)
    



    
![png](README_files/README_31_57.png)
    



    
![png](README_files/README_31_58.png)
    



    
![png](README_files/README_31_59.png)
    



    
![png](README_files/README_31_60.png)
    



    
![png](README_files/README_31_61.png)
    



    
![png](README_files/README_31_62.png)
    



    
![png](README_files/README_31_63.png)
    



    
![png](README_files/README_31_64.png)
    



    
![png](README_files/README_31_65.png)
    



    
![png](README_files/README_31_66.png)
    



    
![png](README_files/README_31_67.png)
    



    
![png](README_files/README_31_68.png)
    



    
![png](README_files/README_31_69.png)
    



    
![png](README_files/README_31_70.png)
    



    
![png](README_files/README_31_71.png)
    



    
![png](README_files/README_31_72.png)
    



    
![png](README_files/README_31_73.png)
    



    
![png](README_files/README_31_74.png)
    



    
![png](README_files/README_31_75.png)
    



    
![png](README_files/README_31_76.png)
    



    
![png](README_files/README_31_77.png)
    



    
![png](README_files/README_31_78.png)
    



    
![png](README_files/README_31_79.png)
    



    
![png](README_files/README_31_80.png)
    



    
![png](README_files/README_31_81.png)
    



    
![png](README_files/README_31_82.png)
    



    
![png](README_files/README_31_83.png)
    



    
![png](README_files/README_31_84.png)
    



    
![png](README_files/README_31_85.png)
    



    
![png](README_files/README_31_86.png)
    



    
![png](README_files/README_31_87.png)
    



    
![png](README_files/README_31_88.png)
    



    
![png](README_files/README_31_89.png)
    



    
![png](README_files/README_31_90.png)
    



    
![png](README_files/README_31_91.png)
    



    
![png](README_files/README_31_92.png)
    



    
![png](README_files/README_31_93.png)
    



    
![png](README_files/README_31_94.png)
    



    
![png](README_files/README_31_95.png)
    



    
![png](README_files/README_31_96.png)
    



    
![png](README_files/README_31_97.png)
    



    
![png](README_files/README_31_98.png)
    



    
![png](README_files/README_31_99.png)
    



    
![png](README_files/README_31_100.png)
    



    
![png](README_files/README_31_101.png)
    



    
![png](README_files/README_31_102.png)
    



    
![png](README_files/README_31_103.png)
    



    
![png](README_files/README_31_104.png)
    



    
![png](README_files/README_31_105.png)
    



    
![png](README_files/README_31_106.png)
    



    
![png](README_files/README_31_107.png)
    



    
![png](README_files/README_31_108.png)
    



    
![png](README_files/README_31_109.png)
    



    
![png](README_files/README_31_110.png)
    



    
![png](README_files/README_31_111.png)
    



    
![png](README_files/README_31_112.png)
    



    
![png](README_files/README_31_113.png)
    



    
![png](README_files/README_31_114.png)
    



    
![png](README_files/README_31_115.png)
    



    
![png](README_files/README_31_116.png)
    



    
![png](README_files/README_31_117.png)
    



    
![png](README_files/README_31_118.png)
    



    
![png](README_files/README_31_119.png)
    



    
![png](README_files/README_31_120.png)
    



    
![png](README_files/README_31_121.png)
    



    
![png](README_files/README_31_122.png)
    



    
![png](README_files/README_31_123.png)
    



    
![png](README_files/README_31_124.png)
    



    
![png](README_files/README_31_125.png)
    



    
![png](README_files/README_31_126.png)
    



    
![png](README_files/README_31_127.png)
    



    
![png](README_files/README_31_128.png)
    



    
![png](README_files/README_31_129.png)
    



    
![png](README_files/README_31_130.png)
    



    
![png](README_files/README_31_131.png)
    



    
![png](README_files/README_31_132.png)
    



    
![png](README_files/README_31_133.png)
    



    
![png](README_files/README_31_134.png)
    



    
![png](README_files/README_31_135.png)
    



    
![png](README_files/README_31_136.png)
    



    
![png](README_files/README_31_137.png)
    



    
![png](README_files/README_31_138.png)
    



    
![png](README_files/README_31_139.png)
    



    
![png](README_files/README_31_140.png)
    



    
![png](README_files/README_31_141.png)
    



    
![png](README_files/README_31_142.png)
    



    
![png](README_files/README_31_143.png)
    



    
![png](README_files/README_31_144.png)
    



    
![png](README_files/README_31_145.png)
    



    
![png](README_files/README_31_146.png)
    



    
![png](README_files/README_31_147.png)
    



    
![png](README_files/README_31_148.png)
    



    
![png](README_files/README_31_149.png)
    



    
![png](README_files/README_31_150.png)
    



    
![png](README_files/README_31_151.png)
    



    
![png](README_files/README_31_152.png)
    



    
![png](README_files/README_31_153.png)
    



    
![png](README_files/README_31_154.png)
    



    
![png](README_files/README_31_155.png)
    



    
![png](README_files/README_31_156.png)
    



    
![png](README_files/README_31_157.png)
    



    
![png](README_files/README_31_158.png)
    



    
![png](README_files/README_31_159.png)
    



    
![png](README_files/README_31_160.png)
    



    
![png](README_files/README_31_161.png)
    



    
![png](README_files/README_31_162.png)
    



    
![png](README_files/README_31_163.png)
    



    
![png](README_files/README_31_164.png)
    



    
![png](README_files/README_31_165.png)
    



    
![png](README_files/README_31_166.png)
    



    
![png](README_files/README_31_167.png)
    



    
![png](README_files/README_31_168.png)
    



    
![png](README_files/README_31_169.png)
    



    
![png](README_files/README_31_170.png)
    



    
![png](README_files/README_31_171.png)
    



    
![png](README_files/README_31_172.png)
    



    
![png](README_files/README_31_173.png)
    



    
![png](README_files/README_31_174.png)
    



    
![png](README_files/README_31_175.png)
    



    
![png](README_files/README_31_176.png)
    



    
![png](README_files/README_31_177.png)
    



    
![png](README_files/README_31_178.png)
    



    
![png](README_files/README_31_179.png)
    



    
![png](README_files/README_31_180.png)
    



    
![png](README_files/README_31_181.png)
    



    
![png](README_files/README_31_182.png)
    



    
![png](README_files/README_31_183.png)
    



    
![png](README_files/README_31_184.png)
    



    
![png](README_files/README_31_185.png)
    



    
![png](README_files/README_31_186.png)
    



    
![png](README_files/README_31_187.png)
    



    
![png](README_files/README_31_188.png)
    



    
![png](README_files/README_31_189.png)
    



    
![png](README_files/README_31_190.png)
    



    
![png](README_files/README_31_191.png)
    



    
![png](README_files/README_31_192.png)
    



    
![png](README_files/README_31_193.png)
    



    
![png](README_files/README_31_194.png)
    



    
![png](README_files/README_31_195.png)
    



    
![png](README_files/README_31_196.png)
    



    
![png](README_files/README_31_197.png)
    



    
![png](README_files/README_31_198.png)
    



    
![png](README_files/README_31_199.png)
    



    
![png](README_files/README_31_200.png)
    



    
![png](README_files/README_31_201.png)
    



    
![png](README_files/README_31_202.png)
    



    
![png](README_files/README_31_203.png)
    



    
![png](README_files/README_31_204.png)
    



    
![png](README_files/README_31_205.png)
    



    
![png](README_files/README_31_206.png)
    



    
![png](README_files/README_31_207.png)
    



    
![png](README_files/README_31_208.png)
    



    
![png](README_files/README_31_209.png)
    



    
![png](README_files/README_31_210.png)
    



    
![png](README_files/README_31_211.png)
    



    
![png](README_files/README_31_212.png)
    



    
![png](README_files/README_31_213.png)
    



    
![png](README_files/README_31_214.png)
    



    
![png](README_files/README_31_215.png)
    


### Continuous-to-Continuous Relationships or Conditional Distributions: Correlation Plots
* The correlation matrix reveals relationships between health conditions, lifestyle factors, and demographic attributes, with a strong positive correlation between height and weight (0.47), both also moderately associated with BMI (weight at 0.40). Depressive disorders show correlations with difficulty concentrating (0.34), difficulty walking (0.24), and difficulty with errands (0.23), highlighting the link between mental and physical limitations. Chronic conditions like arthritis, asthma, and COPD correlate with physical difficulties, such as arthritis with difficulty walking (0.32) and difficulty dressing (0.38), indicating that these conditions impact daily activities. Depressive disorders and kidney disease have a moderate correlation (0.20), suggesting a potential link between mental health and overall health decline. Preventive health behaviors, including HIV testing and flu vaccination, show low correlations with most health conditions, suggesting these are applied broadly across health statuses. Alcohol consumption has minimal correlations with other variables, indicating no significant associations with health conditions or physical metrics. Overall, the matrix underscores significant associations between mental and physical health challenges and daily functional difficulties, particularly with chronic conditions affecting mobility and daily tasks, while preventive measures and lifestyle factors like vaccination and alcohol use show independence from these health conditions.


```python
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(data=df.drop(columns=[target]).\
            corr(numeric_only=True),
            vmin=-1, 
            vmax=1, 
            center=0,
            cbar=False,
            annot=True, 
            annot_kws={'size': 7},
            fmt='.2f',
            ax=ax)
plt.show()
```


    
![png](README_files/README_33_0.png)
    


## Binary Classification Model

### Formulas

#### Formula 1: Additive Terms Only


```python
formula_additive = """
    HadHeartAttack ~ 
    C(State) + 
    C(Sex) + 
    C(GeneralHealth) + 
    C(AgeCategory) + 
    C(HadDiabetes) + 
    C(SmokerStatus) + 
    C(ECigaretteUsage) + 
    C(RaceEthnicityCategory) + 
    C(TetanusLast10Tdap) + 
    HeightInMeters + 
    WeightInKilograms + 
    BMI + 
    HadAngina + 
    HadStroke + 
    HadAsthma + 
    HadSkinCancer + 
    HadCOPD + 
    HadDepressiveDisorder + 
    HadKidneyDisease + 
    HadArthritis + 
    DeafOrHardOfHearing + 
    BlindOrVisionDifficulty + 
    DifficultyConcentrating + 
    DifficultyWalking + 
    DifficultyDressingBathing + 
    DifficultyErrands + 
    ChestScan + 
    AlcoholDrinkers + 
    HIVTesting + 
    FluVaxLast12 + 
    PneumoVaxEver + 
    HighRiskLastYear + 
    CovidPos
"""
```

#### Formula 2: Additive and Interaction Terms


```python
formula_additive_and_interactive = """
    HadHeartAttack ~
    C(State) +
    C(Sex) * C(AgeCategory) +
    C(GeneralHealth) +
    C(HadDiabetes) * C(SmokerStatus) +
    C(ECigaretteUsage) +
    C(RaceEthnicityCategory) +
    C(TetanusLast10Tdap) +
    HeightInMeters +
    WeightInKilograms +
    BMI +
    HadAngina +
    HadStroke +
    HadAsthma +
    HadSkinCancer +
    HadCOPD +
    HadDepressiveDisorder +
    HadKidneyDisease +
    HadArthritis +
    DeafOrHardOfHearing +
    BlindOrVisionDifficulty +
    DifficultyConcentrating +
    DifficultyWalking +
    DifficultyDressingBathing +
    DifficultyErrands +
    ChestScan +
    AlcoholDrinkers +
    HIVTesting +
    FluVaxLast12 +
    PneumoVaxEver +
    HighRiskLastYear +
    CovidPos
"""
```

#### Combined Formulas


```python
formulas = [formula_additive,
            formula_additive_and_interactive]
```

### Apply 5-Fold Cross-Validation


```python
from sklearn.model_selection import StratifiedKFold
```


```python
kf = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=101)
```

### Fit the Logistic Regression Models w/ Statsmodels


```python
input_names = df.drop(columns=[target]).\
                      copy().\
                      columns.\
                      to_list()
```


```python
output_name = target
```


```python
from sklearn.preprocessing import StandardScaler
```


```python
from sklearn.metrics import roc_auc_score
```


```python
def my_coefplot(model, figsize_default=(10, 4), figsize_expansion_factor=0.5, max_default_vars=10):
    """
    Function that plots a coefficient plot with error bars for a given statistical model
    and prints out which variables are statistically significant and whether they are positive or negative.
    The graph height dynamically adjusts based on the number of variables.

    Params:
        model: object
        figsize_default: tuple, optional
        figsize_expansion_factor: float, optional
        max_default_vars: int, optional
    """
    # cap the standard errors (bse) to avoid overly large error bars, upper bound set to 2
    capped_bse = model.bse.clip(upper=2)
    
    # calculate the minimum and maximum coefficient values adjusted by the standard errors
    coef_min = (model.params - 2 * capped_bse).min()
    coef_max = (model.params + 2 * capped_bse).max()
    
    # define buffer space for the x-axis limits
    buffer_space = 0.5
    xlim_min = coef_min - buffer_space
    xlim_max = coef_max + buffer_space
    
    # dynamically calculate figure height based on the number of variables
    num_vars = len(model.params)
    if num_vars > max_default_vars:
        height = figsize_default[1] + figsize_expansion_factor * (num_vars - max_default_vars)
    else:
        height = figsize_default[1]
    
    # create the plot
    fig, ax = plt.subplots(figsize=(figsize_default[0], height))
    
    # identify statistically significant and non-significant variables based on p-values
    significant_vars = model.pvalues[model.pvalues < 0.05].index
    not_significant_vars = model.pvalues[model.pvalues >= 0.05].index
    
    # plot non-significant variables with grey error bars
    ax.errorbar(y=not_significant_vars,
                x=model.params[not_significant_vars],
                xerr=2 * capped_bse[not_significant_vars],
                fmt='o', 
                color='grey', 
                ecolor='grey', 
                elinewidth=2, 
                ms=10,
                label='not significant')
    
    # plot significant variables with red error bars
    ax.errorbar(y=significant_vars,
                x=model.params[significant_vars],
                xerr=2 * capped_bse[significant_vars],
                fmt='o', 
                color='red', 
                ecolor='red', 
                elinewidth=2, 
                ms=10,
                label='significant (p < 0.05)')
    
    # add a vertical line at 0 to visually separate positive and negative coefficients
    ax.axvline(x=0, linestyle='--', linewidth=2.5, color='grey')
    
    # adjust the x-axis limits to add some buffer space on either side
    ax.set_xlim(min(-0.5, coef_min - 0.2), max(0.5, coef_max + 0.2))
    ax.set_xlabel('coefficient value')
    
    # add legend to distinguish between significant and non-significant variables
    ax.legend()
    
    # show the plot
    plt.show()
    
    # print the summary of statistically significant variables
    print('\n--- statistically significant variables ---')
    
    # check if there are any significant variables, if not, print a message
    if significant_vars.empty:
        print('No statistically significant variables found.')
    else:
        # for each significant variable, print its coefficient, standard error, p-value, and direction
        for var in significant_vars:
            coef_value = model.params[var]
            std_err = model.bse[var]
            p_val = model.pvalues[var]
            direction = 'positive' if coef_value > 0 else 'negative'
            print(f'variable: {var}, coefficient: {coef_value:.4f}, std err: {std_err:.4f}, p-value: {p_val:.4f}, direction: {direction}')
```


```python
def train_and_test_logistic_with_cv(model, formula, df, x_names, y_name, cv, threshold=0.5, use_scaler=True):
    """
    Function to train and test a logistic binary classification model with Cross-Validation,
    including accuracy and ROC AUC score calculations.

    Params:
        model: object
        formula: str
        df: object
        x_names: list
        y_name: str
        cv: object
        threshold: float, optional
        use_scaler: bool, optional

    Returns:
        object
    """
    # separate the inputs and output
    input_df = df.loc[:, x_names].copy()
    
    # initialize the performance metric storage lists
    train_res = []
    test_res = []
    train_auc_scores = []
    test_auc_scores = []
    
    # split the data and iterate over the folds
    for train_id, test_id in cv.split(input_df.to_numpy(), df[y_name].to_numpy()):
        
        # subset the training and test splits within each fold
        train_data = df.iloc[train_id, :].copy()
        test_data = df.iloc[test_id, :].copy()

        # if the use_scaler flag is set, standardize the numeric features within each fold
        if use_scaler:
            scaler = StandardScaler()
            
            # identify numeric columns to scale, excluding the target variable
            columns_to_scale = train_data.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_scale = [col for col in columns_to_scale if col != y_name]
            
            # fit scaler on training data
            scaler.fit(train_data[columns_to_scale])
            
            # transform training and test data
            train_data[columns_to_scale] = scaler.transform(train_data[columns_to_scale])
            test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
        
        # fit the model on the training data within the current fold
        a_model = smf.logit(formula=formula, data=train_data).fit()
        
        # predict the training within each fold
        train_copy = train_data.copy()
        train_copy['pred_probability'] = a_model.predict(train_data)
        train_copy['pred_class'] = np.where(train_copy.pred_probability > threshold, 1, 0)
        
        # predict the testing within each fold
        test_copy = test_data.copy()
        test_copy['pred_probability'] = a_model.predict(test_data)
        test_copy['pred_class'] = np.where(test_copy.pred_probability > threshold, 1, 0)
        
        # calculate the performance metric (accuracy) on the training set within the fold
        train_res.append(np.mean(train_copy[y_name] == train_copy.pred_class))
        
        # calculate the performance metric (accuracy) on the testing set within the fold
        test_res.append(np.mean(test_copy[y_name] == test_copy.pred_class))

        # calculate the roc_auc_score for the training set
        train_auc_scores.append(roc_auc_score(train_copy[y_name], train_copy['pred_probability']))
        
        # calculate the roc_auc_score for the testing_set
        test_auc_scores.append(roc_auc_score(test_copy[y_name], test_copy['pred_probability']))
    
    # book keeping to store the results (accuracy)
    train_df = pd.DataFrame({'accuracy': train_res, 'roc_auc': train_auc_scores})
    train_df['from_set'] = 'training'
    train_df['fold_id'] = train_df.index + 1
    test_df = pd.DataFrame({'accuracy': test_res, 'roc_auc': test_auc_scores})
    test_df['from_set'] = 'testing'
    test_df['fold_id'] = test_df.index + 1
    
    # combine the splits together
    res_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # add information about the model
    res_df['model'] = model
    res_df['formula'] = formula
    res_df['num_coefs'] = len(a_model.params)
    res_df['threshold'] = threshold

    # return the results DataFrame
    return res_df
```

### Test Models


```python
import os
```


```python
import contextlib
```


```python
res_list = []
error_log = []

with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
    for model in range(len(formulas)):
        try:
            res_list.append(train_and_test_logistic_with_cv(model,
                                                            formula=formulas[model],
                                                            df=df,
                                                            x_names=input_names,
                                                            y_name=output_name,
                                                            cv=kf))
        except Exception as e:
            error_log.append(f'Formula ID {model} failed: {str(e)}')
```


```python
cv_results = pd.concat(res_list, ignore_index=True)
```


```python
cv_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>roc_auc</th>
      <th>from_set</th>
      <th>fold_id</th>
      <th>model</th>
      <th>formula</th>
      <th>num_coefs</th>
      <th>threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.947802</td>
      <td>0.888335</td>
      <td>training</td>
      <td>1</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.947592</td>
      <td>0.888224</td>
      <td>training</td>
      <td>2</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.947665</td>
      <td>0.887070</td>
      <td>training</td>
      <td>3</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.947534</td>
      <td>0.886954</td>
      <td>training</td>
      <td>4</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.947481</td>
      <td>0.886485</td>
      <td>training</td>
      <td>5</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.947229</td>
      <td>0.881833</td>
      <td>testing</td>
      <td>1</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.947587</td>
      <td>0.883013</td>
      <td>testing</td>
      <td>2</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.947292</td>
      <td>0.886866</td>
      <td>testing</td>
      <td>3</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.946892</td>
      <td>0.888745</td>
      <td>testing</td>
      <td>4</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.947860</td>
      <td>0.889945</td>
      <td>testing</td>
      <td>5</td>
      <td>0</td>
      <td>\n    HadHeartAttack ~ \n    C(State) + \n    ...</td>
      <td>111</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.947823</td>
      <td>0.888659</td>
      <td>training</td>
      <td>1</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.947597</td>
      <td>0.888564</td>
      <td>training</td>
      <td>2</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.947729</td>
      <td>0.887387</td>
      <td>training</td>
      <td>3</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.947502</td>
      <td>0.887271</td>
      <td>training</td>
      <td>4</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.947597</td>
      <td>0.886813</td>
      <td>training</td>
      <td>5</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.947187</td>
      <td>0.881816</td>
      <td>testing</td>
      <td>1</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.947439</td>
      <td>0.882991</td>
      <td>testing</td>
      <td>2</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.947292</td>
      <td>0.887122</td>
      <td>testing</td>
      <td>3</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.947208</td>
      <td>0.888707</td>
      <td>testing</td>
      <td>4</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.948113</td>
      <td>0.889904</td>
      <td>testing</td>
      <td>5</td>
      <td>1</td>
      <td>\n    HadHeartAttack ~\n    C(State) +\n    C(...</td>
      <td>132</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



### Review Model Results - Average of Folds

#### All Models w/ Highest Accuracy on the Testing Dataset


```python
cv_results.loc[(cv_results['from_set'] == 'testing') & 
               (cv_results['accuracy'] < 1.0) & 
               (cv_results['roc_auc'] < 1.0)].\
           groupby('model').\
           aggregate({'accuracy': 'mean', 
                      'roc_auc': 'mean', 
                      'num_coefs': 'first'}).\
           reset_index().\
           sort_values(by='accuracy', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy</th>
      <th>roc_auc</th>
      <th>num_coefs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.947448</td>
      <td>0.886108</td>
      <td>132</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.947372</td>
      <td>0.886080</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>



### Model Selection
* The best logistic regression model achieves an accuracy of 0.9474 and an ROC AUC of 0.8861, with 111 coefficients, indicating strong predictive performance and robustness across multiple predictors. The intercept is highly negative (-6.1308, p < 0.0001), suggesting a low baseline probability. Among states, South Dakota (0.5115, p = 0.0001), Maine (0.4423, p = 0.0004), and several others show positive coefficients, reflecting higher probabilities in these locations. Age is a strong positive predictor, with coefficients increasing progressively for older groups, notably ages 80+ (2.4317, p < 0.0001) and 75-79 (2.2763, p < 0.0001), underscoring age as a crucial factor. General health ratings also correlate, with “Poor” health having a substantial positive effect (1.0895, p < 0.0001), followed by “Fair” (0.9304, p < 0.0001). Health conditions like angina (2.4228, p < 0.0001), stroke (0.8598, p < 0.0001), and diabetes (0.3343, p < 0.0001) contribute positively, highlighting their association with the outcome. Difficulty-related variables, such as difficulty walking (0.0897, p = 0.0015) and concentrating (0.0697, p = 0.0346), show moderate positive correlations. Smoking status has a negative impact, with “Former smoker” (-0.2530, p < 0.0001) and “Never smoked” (-0.5112, p < 0.0001) associated with lower probabilities, while e-cigarette usage “Not at all (right now)” (0.0570, p = 0.0484) has a positive influence. Preventive measures show mixed effects, with flu vaccination last year (-0.1420, p < 0.0001) and tetanus shot (Tdap, -0.1020, p = 0.0006) negatively correlated, while pneumococcal vaccine (0.0846, p = 0.0010) is positive. Health behaviors, like chest scans (0.5830, p < 0.0001), indicate an association, while alcohol consumption shows a negative correlation (-0.2104, p < 0.0001). This model suggests that older age, poor health, and specific chronic conditions or disabilities strongly increase the likelihood of the outcome, with some geographic and lifestyle factors contributing as well.


```python
best_model = smf.logit(formula=formulas[0], 
                       data=df).fit()
```

    Optimization terminated successfully.
             Current function value: 0.147331
             Iterations 10



```python
best_model.params
```




    Intercept                -6.130787
    C(State)[T.Alaska]        0.265161
    C(State)[T.Arizona]       0.310066
    C(State)[T.Arkansas]      0.253471
    C(State)[T.California]    0.189936
                                ...   
    HIVTesting                0.065535
    FluVaxLast12             -0.142049
    PneumoVaxEver             0.084587
    HighRiskLastYear          0.100571
    CovidPos                  0.027686
    Length: 111, dtype: float64




```python
best_model.pvalues < 0.05
```




    Intercept                  True
    C(State)[T.Alaska]        False
    C(State)[T.Arizona]        True
    C(State)[T.Arkansas]      False
    C(State)[T.California]    False
                              ...  
    HIVTesting                 True
    FluVaxLast12               True
    PneumoVaxEver              True
    HighRiskLastYear          False
    CovidPos                  False
    Length: 111, dtype: bool




```python
best_model.params[best_model.pvalues < 0.05].sort_values(ascending=False)
```




    C(AgeCategory)[T.Age 80 or older]                            2.431717
    HadAngina                                                    2.422814
    C(AgeCategory)[T.Age 75 to 79]                               2.276268
    C(AgeCategory)[T.Age 70 to 74]                               2.150908
    C(AgeCategory)[T.Age 65 to 69]                               2.036877
    C(AgeCategory)[T.Age 60 to 64]                               1.869937
    C(AgeCategory)[T.Age 55 to 59]                               1.779459
    C(AgeCategory)[T.Age 50 to 54]                               1.572669
    C(AgeCategory)[T.Age 45 to 49]                               1.383461
    C(GeneralHealth)[T.Poor]                                     1.089501
    C(GeneralHealth)[T.Fair]                                     0.930421
    C(AgeCategory)[T.Age 40 to 44]                               0.906325
    HadStroke                                                    0.859761
    C(AgeCategory)[T.Age 35 to 39]                               0.732504
    C(Sex)[T.Male]                                               0.696715
    C(GeneralHealth)[T.Good]                                     0.680010
    ChestScan                                                    0.583000
    C(State)[T.South Dakota]                                     0.511517
    C(State)[T.Maine]                                            0.442309
    C(AgeCategory)[T.Age 30 to 34]                               0.423445
    C(RaceEthnicityCategory)[T.Multiracial, Non-Hispanic]        0.407343
    C(State)[T.New Hampshire]                                    0.341477
    C(GeneralHealth)[T.Very good]                                0.338416
    C(HadDiabetes)[T.Yes]                                        0.334325
    C(HadDiabetes)[T.Yes, but only during pregnancy (female)]    0.333506
    C(State)[T.Nebraska]                                         0.332872
    C(State)[T.Idaho]                                            0.328569
    C(State)[T.Colorado]                                         0.324521
    C(State)[T.Vermont]                                          0.311294
    C(State)[T.Massachusetts]                                    0.310719
    C(State)[T.Arizona]                                          0.310066
    C(RaceEthnicityCategory)[T.Other race only, Non-Hispanic]    0.309601
    C(State)[T.Montana]                                          0.292848
    C(State)[T.New Mexico]                                       0.280003
    C(State)[T.Ohio]                                             0.277925
    C(State)[T.Florida]                                          0.265763
    C(RaceEthnicityCategory)[T.Hispanic]                         0.207655
    BlindOrVisionDifficulty                                      0.132919
    C(RaceEthnicityCategory)[T.White only, Non-Hispanic]         0.124063
    DifficultyWalking                                            0.089650
    DifficultyErrands                                            0.089172
    PneumoVaxEver                                                0.084587
    HadKidneyDisease                                             0.069949
    DifficultyConcentrating                                      0.069724
    HIVTesting                                                   0.065535
    C(ECigaretteUsage)[T.Not at all (right now)]                 0.057042
    C(TetanusLast10Tdap)[T.Yes, received Tdap]                  -0.101976
    HadSkinCancer                                               -0.124903
    FluVaxLast12                                                -0.142049
    AlcoholDrinkers                                             -0.210428
    C(SmokerStatus)[T.Former smoker]                            -0.253021
    C(SmokerStatus)[T.Never smoked]                             -0.511246
    Intercept                                                   -6.130787
    dtype: float64




```python
my_coefplot(best_model)
```


    
![png](README_files/README_66_0.png)
    


    
    --- statistically significant variables ---
    variable: Intercept, coefficient: -6.1308, std err: 0.7241, p-value: 0.0000, direction: negative
    variable: C(State)[T.Arizona], coefficient: 0.3101, std err: 0.1275, p-value: 0.0150, direction: positive
    variable: C(State)[T.Colorado], coefficient: 0.3245, std err: 0.1362, p-value: 0.0172, direction: positive
    variable: C(State)[T.Florida], coefficient: 0.2658, std err: 0.1220, p-value: 0.0294, direction: positive
    variable: C(State)[T.Idaho], coefficient: 0.3286, std err: 0.1433, p-value: 0.0219, direction: positive
    variable: C(State)[T.Maine], coefficient: 0.4423, std err: 0.1259, p-value: 0.0004, direction: positive
    variable: C(State)[T.Massachusetts], coefficient: 0.3107, std err: 0.1338, p-value: 0.0202, direction: positive
    variable: C(State)[T.Montana], coefficient: 0.2928, std err: 0.1353, p-value: 0.0304, direction: positive
    variable: C(State)[T.Nebraska], coefficient: 0.3329, std err: 0.1288, p-value: 0.0098, direction: positive
    variable: C(State)[T.New Hampshire], coefficient: 0.3415, std err: 0.1354, p-value: 0.0117, direction: positive
    variable: C(State)[T.New Mexico], coefficient: 0.2800, std err: 0.1416, p-value: 0.0480, direction: positive
    variable: C(State)[T.Ohio], coefficient: 0.2779, std err: 0.1211, p-value: 0.0217, direction: positive
    variable: C(State)[T.South Dakota], coefficient: 0.5115, std err: 0.1314, p-value: 0.0001, direction: positive
    variable: C(State)[T.Vermont], coefficient: 0.3113, std err: 0.1341, p-value: 0.0202, direction: positive
    variable: C(Sex)[T.Male], coefficient: 0.6967, std err: 0.0309, p-value: 0.0000, direction: positive
    variable: C(GeneralHealth)[T.Fair], coefficient: 0.9304, std err: 0.0513, p-value: 0.0000, direction: positive
    variable: C(GeneralHealth)[T.Good], coefficient: 0.6800, std err: 0.0478, p-value: 0.0000, direction: positive
    variable: C(GeneralHealth)[T.Poor], coefficient: 1.0895, std err: 0.0595, p-value: 0.0000, direction: positive
    variable: C(GeneralHealth)[T.Very good], coefficient: 0.3384, std err: 0.0490, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 30 to 34], coefficient: 0.4234, std err: 0.1789, p-value: 0.0180, direction: positive
    variable: C(AgeCategory)[T.Age 35 to 39], coefficient: 0.7325, std err: 0.1658, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 40 to 44], coefficient: 0.9063, std err: 0.1600, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 45 to 49], coefficient: 1.3835, std err: 0.1538, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 50 to 54], coefficient: 1.5727, std err: 0.1507, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 55 to 59], coefficient: 1.7795, std err: 0.1491, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 60 to 64], coefficient: 1.8699, std err: 0.1484, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 65 to 69], coefficient: 2.0369, std err: 0.1482, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 70 to 74], coefficient: 2.1509, std err: 0.1486, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 75 to 79], coefficient: 2.2763, std err: 0.1496, p-value: 0.0000, direction: positive
    variable: C(AgeCategory)[T.Age 80 or older], coefficient: 2.4317, std err: 0.1498, p-value: 0.0000, direction: positive
    variable: C(HadDiabetes)[T.Yes], coefficient: 0.3343, std err: 0.0249, p-value: 0.0000, direction: positive
    variable: C(HadDiabetes)[T.Yes, but only during pregnancy (female)], coefficient: 0.3335, std err: 0.1493, p-value: 0.0255, direction: positive
    variable: C(SmokerStatus)[T.Former smoker], coefficient: -0.2530, std err: 0.0359, p-value: 0.0000, direction: negative
    variable: C(SmokerStatus)[T.Never smoked], coefficient: -0.5112, std err: 0.0368, p-value: 0.0000, direction: negative
    variable: C(ECigaretteUsage)[T.Not at all (right now)], coefficient: 0.0570, std err: 0.0289, p-value: 0.0484, direction: positive
    variable: C(RaceEthnicityCategory)[T.Hispanic], coefficient: 0.2077, std err: 0.0627, p-value: 0.0009, direction: positive
    variable: C(RaceEthnicityCategory)[T.Multiracial, Non-Hispanic], coefficient: 0.4073, std err: 0.0811, p-value: 0.0000, direction: positive
    variable: C(RaceEthnicityCategory)[T.Other race only, Non-Hispanic], coefficient: 0.3096, std err: 0.0691, p-value: 0.0000, direction: positive
    variable: C(RaceEthnicityCategory)[T.White only, Non-Hispanic], coefficient: 0.1241, std err: 0.0440, p-value: 0.0049, direction: positive
    variable: C(TetanusLast10Tdap)[T.Yes, received Tdap], coefficient: -0.1020, std err: 0.0298, p-value: 0.0006, direction: negative
    variable: HadAngina, coefficient: 2.4228, std err: 0.0232, p-value: 0.0000, direction: positive
    variable: HadStroke, coefficient: 0.8598, std err: 0.0314, p-value: 0.0000, direction: positive
    variable: HadSkinCancer, coefficient: -0.1249, std err: 0.0323, p-value: 0.0001, direction: negative
    variable: HadKidneyDisease, coefficient: 0.0699, std err: 0.0345, p-value: 0.0426, direction: positive
    variable: BlindOrVisionDifficulty, coefficient: 0.1329, std err: 0.0371, p-value: 0.0003, direction: positive
    variable: DifficultyConcentrating, coefficient: 0.0697, std err: 0.0330, p-value: 0.0346, direction: positive
    variable: DifficultyWalking, coefficient: 0.0897, std err: 0.0282, p-value: 0.0015, direction: positive
    variable: DifficultyErrands, coefficient: 0.0892, std err: 0.0371, p-value: 0.0163, direction: positive
    variable: ChestScan, coefficient: 0.5830, std err: 0.0246, p-value: 0.0000, direction: positive
    variable: AlcoholDrinkers, coefficient: -0.2104, std err: 0.0224, p-value: 0.0000, direction: negative
    variable: HIVTesting, coefficient: 0.0655, std err: 0.0249, p-value: 0.0084, direction: positive
    variable: FluVaxLast12, coefficient: -0.1420, std err: 0.0237, p-value: 0.0000, direction: negative
    variable: PneumoVaxEver, coefficient: 0.0846, std err: 0.0258, p-value: 0.0010, direction: positive


### Save Model - Usage in Inference Application


```python
import pickle
```


```python
with open('patients_data_logit_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
```

### Load Model - Usage in Inference Application


```python
with open('patients_data_logit_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

### Perform Inference - Validate Production Model


```python
sample_data = pd.DataFrame({
    'State': ['California'],
    'Sex': ['Male'],
    'GeneralHealth': ['Good'],
    'AgeCategory': ['Age 18 to 24'],
    'HeightInMeters': [1.75],
    'WeightInKilograms': [70.0],
    'BMI': [22.86],
    'HadHeartAttack': [0],
    'HadAngina': [0],
    'HadStroke': [0],
    'HadAsthma': [1],
    'HadSkinCancer': [0],
    'HadCOPD': [0],
    'HadDepressiveDisorder': [0],
    'HadKidneyDisease': [0],
    'HadArthritis': [0],
    'HadDiabetes': ['No'],
    'DeafOrHardOfHearing': [0],
    'BlindOrVisionDifficulty': [0],
    'DifficultyConcentrating': [0],
    'DifficultyWalking': [0],
    'DifficultyDressingBathing': [0],
    'DifficultyErrands': [0],
    'SmokerStatus': ['Never smoked'],
    'ECigaretteUsage': ['Not at all (right now)'],
    'ChestScan': [0],
    'RaceEthnicityCategory': ['White only, Non-Hispanic'],
    'AlcoholDrinkers': [1],
    'HIVTesting': [0],
    'FluVaxLast12': [1],
    'PneumoVaxEver': [0],
    'TetanusLast10Tdap': ['Yes, received Tdap'],
    'HighRiskLastYear': [0],
    'CovidPos': [0]
})
```


```python
loaded_model.predict(sample_data)
```




    0    0.003134
    dtype: float64


