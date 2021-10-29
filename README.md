# Bird Classification Project

Contributors: Crystal Gould Perrott, Jeff Marvel

## Business Problem

In September 2021, the Ivory-billed Woodpecker was declared extinct. Eight species of birds in Hawaii are expected to share a similar fate due to habitat removal and the presence of invasive species. The World Ornithological Society (WOS) has an interest in the conservation of bird species of the world. To best direct research funding and target conservation efforts the WOS has asked us to build a predictive model to accurately identify threatened bird species.
<br />  
We aim to use Machine Learning techniques to construct a predictive model trained on current threatened status. Our data includes diet, strategies of foraging/hunting, biological features of the bird(s), types of environmental threats, habitats, and breeding location.  
<br />
Developing a model to predict threatened status, we can identify factors that most contribute to species decline. The WOS can utilize the model to strategically plan conservation efforts and prioritize research funding. 
<br />  
We will take a particular interest in minimizing false negatives, i.e., where our model has not predicted a bird is threatened, when in reality it is. Mis-identifying a bird as "not threatened" is the worst case scenario for the WOS. False positives on the other hand are not particularly impactful. The "worst" case here is a slightly misguided conservation approach, which will have minimal harm. E.g., conservation efforts like protecting habitat and reducing pollution are still impactful even if theyy slightly miss the mark.

## Navigating the repository
* The final notebook is bird_conservation_notebook.ipynb in the main folder. It includes the main modeling iterations as well as our final model.
* Datasets used are bird_dataset.csv and df.csv (for later modeling iterations) in the "Data" folder
* Some model iterations such as grid searches were not included in the final notebook. Notebooks referenced in the final analysis can be found in the "Notebooks" folder under the relevant name
* util.py in the main folder contains functions for evaluating classification model results as well as measuring multicollinearity in the dataset. Refer to the docstring for full details.

## Data
Data was combined from two sources and combined by matching on Scientific or English name
* The first source is a research paper published in the Ecological Society of America. For most bird species, the researchers collected diet information, foraging (feeding) strategy, and other characteristics about the bird (weight, nocturnal, passerine)
* The second was BirdLife.org, a non profit focused on bird conservation. Their data zone includes information on bird habitat, region, and the nature of the threat these species face (e.g., agriculture, climate change, etc).

Diet Data Source:
* Paper: https://figshare.com/collections/EltonTraits_1_0_Species-level_foraging_attributes_of_the_world_s_birds_and_mammals/3306933

Other bird Feature Source:
* Data Dictionary: http://datazone.birdlife.org/species/spcrange
* Data Query Portal: http://datazone.birdlife.org/species/search
* Main Website: https://www.birdlife.org  

Red List (threatened, endangered, etc):
* https://www.iucnredlist.org

## Exploratory Data Analysis
The final data includes records on 9,597 species, of which ~13% (1,200+) are considered Globally Threatened (Vulnerable, Endangered, or Critically Endangered). 
<br />
Since most variables are binary flags, and the dataset did not include nulls, there were only a small number of data cleaning steps needed:
* Removing a few extinct species and where threatened status was "Data Deficient"
* Remove size outliers (think: large, flightless birds like an Ostrich or Emu)
* One hot encoded a few categorical variables

We engineered some additional fields based on intuition and early model results:
* Total number of threats faced
* Total number of habitats
* Total number of regions
* Agricultural threat where the species only relied on a single habitat
  
## Modeling Approach

Given that the business problem calls for inferential approach, several classification models, including KNN and Bayes were not appropriate for this problem. The first models used Logistic Regression, producing decent results. The precision and recall could be tuned using various threshold, however, this often resulted in an unnacceptable loss in accuracy. Since there was high multi-collinearity and non-normality among many of the features, we decided a decision-tree based approach would be more suitable. We tried basic SKlearn decision tree classifiers, random forests, and XGboost, before settling on Catboost for the final model. Given the high number of categorical features in our data, Catboost produced the best accuracy while controlling for high recall (minimizing false negatives).
  
## Final Model and Recommendations

Using Catboost and tuning certain paremeters, the final model produces an accuracy of 90.16% and recall of 92.26%. On the test data, it only missed 24 species that are actually threatened. The variables that resulted in the most meaningful decision tree splits were Number of Threats, Agriculture and Invasive Species threats, Region (specifically Europe, Central America, and Asia), and Endemic Breeding. Our business recommendations are built on these feature importances:
* Preserve habitat at risk of agriculture or resource use expansion and focus on eradicating invasive species
* Focus conservation efforts on Oceania and High Seas, which have the highest incidence of threatened species. Europe, Central America, and Asia were strong predictors of Not Threatened status, suggesting resources can be shifted away from these regions.
* Endemic breeders (species that are only located in a single country when breeding) are particularly at risk for being threatened and could benefit from additional conservation efforts.
  
## Conclusion and Next Steps

While we were able to improve on baseline accuracy and not miss many endangered species, it still produced a number of false positives. Regardless, the model highlighted important features that can be used to direct conservation efforts. Next steps could include:
* A temporal analysis to model changes in threatened status over time
* Incorporate additional data such as observations through time to help improve model accuracy.
  
## Presentation
  
https://docs.google.com/presentation/d/1-RFhnR5_VmIVRvy919HCM-cwzgbX7Vhs5TUlUXY8B8A/edit?usp=sharing
