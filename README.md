# **Data Science : Online Course from Gonie** 
Course homepage for "Data Science" @ Gonie Ahn
Cover Data Scientist of low level to high level
## Notice
* [**Syllabus**](https://github.com/GonieAhn/Data-Science-online-course-from-gonie/blob/main/00_Syllabus.pdf)
    * 보안 ISSUE로 Data Preprocessing with Python는 Class에서 제외
* **Course Owner  : [Gonie Ahn](https://github.com/GonieAhn)**
    * E) : gonie.ahn@gmail.com
* [**Course History**](https://github.com/GonieAhn/Data-Science-online-course-from-gonie/blob/main/99_Album.pdf)
## Contents
### Data Store - [[Toy Data]](http://mo.lge.com/hub/smartdata/data-science-online-course-from-dstp/tree/master/Data%20Store)
* 현재 Github의 경우 Data Upload size를 10MB로 제한하였기 때문에 small data를 Upload 함
    * .csv 형태
### [Class01] Introduction to Data Analytics - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass01%5D%20Introduction%20to%20Data%20Analytics/C0705_Introduction_to_Data_Analytics.pdf)
* Data Analytics에 대한 전반적인 내용
    * Keyword : *#Data Analytics #Data Science*
### [Class02] Data Loading from AWS(S3) - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass02%5D%20Data%20Loading%20from%20AWS(S3)/C0705_Data_Loading_from_AWS_S3_.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass02%5D%20Data%20Loading%20from%20AWS(S3)/Data_Load_Single.ipynb)
* 빅데이터실 분석서버에 가상환경 만드는 방
* AWS 클라우드 Burket인 S3에서 Python 분석환경으로 데이터 Load 하는 법
* Partitioning 되어 있는 File들을 Multiprocessing을 사용하여 빠르게 불러옴
    * Keyword : *#AWS #S3 #Multiprocessing #pickle #Virtual Environment*
```
# Install Package
- conda install -c anaconda boto3 
- conda install -c conda-forge datatable
- conda install -c conda-forge tqdm
```
### [Class03] Basic of Data Analytics - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass04%5D%20Basic%20of%20Data%20Analytics/C0706_Basic_of_Data_Analytics.pdf)
* 데이터 분석에 앞서 필요한 전반적인 지식
* 데이터 종류와 변수의 종류 정의
* Regression & Classification에 대한 정의
    * Keyword : *#Bias VS Variance #Overfitting VS Underfitting #Loss Function #K-fold Cross Validation*
### [Class04] Regression Problem - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass05%5D%20Regression%20Problem/C0706_Regression_Problem.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass05%5D%20Regression%20Problem/Regression_Problem.ipynb)
* Regression Loss Function
* Regression Model 평가 및 지표해석
* 데이터 실습
* 고려대학교 DMQA Lab. 김성범 교수님 강의 자료를 참고함
    * Keyword : *#Linear Regression #R2 #MSE*
```
# Install Package
- pip install regressors
    - Do not Support Anaconda
```
### [Class05] Regularized Linear Models - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass06%5D%20Regularized%20Linear%20Models/C0706_Regularized_Linear_Models.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass06%5D%20Regularized%20Linear%20Models/Regularized_Linear_Models.ipynb)
* Feature Selection 기법 중 Embedded 기법을 활용함
* 계수 Penalty Term을 주어 분석에 필요하고 중요한 변수만 선택되게 하는 기법
* 데이터 실습
* 고려대학교 DMQA Lab. 김성범 교수님 강의 자료를 참고함
    * Keyword : *#Ridge #LASSO # ElasticNet*
### [Class06] Classification Problem - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass07%5D%20Classification%20Problem/C0707_Classification_Problem.pdf), [[Toturial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass07%5D%20Classification%20Problem/DecisionTree.ipynb)
* Classification Loss Function
* Classification Model 평가 및 지표해석
* 데이터 실습
    * Keyword : *#DecisionTree #ACC #Recall #Precision # F1-score #RuleExtraction*
### [Class07] Ensemble Learning - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass08%5D%20Ensemble%20Learning/C0707_Ensemble_Learning.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass08%5D%20Ensemble%20Learning/Ensemble_Learning.ipynb)
* Ensemble의 정의 및 single model보다 좋은 이유
* Bagging, Boosting, Stacking에 대한 정보
* 데이터 실습
* 고려대학교 DSBA Lab. 강필성 교수님 강의 자료를 참고함
    * Keyword : *#RandomForest #Adaboost #Feature Importance Score*
### [Class08] Gradient Boosting Machine(GBM) Family - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass09%5D%20Gradient%20Boosting%20Machine(GBM)%20Family/C0707_Gradient_Boosting_Machine_GBM__Family.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/tree/master/%5BClass09%5D%20Gradient%20Boosting%20Machine(GBM)%20Family/GBM%20Family%20Tutorial%20Code)
* Gradient Boosting Machine 개념 설명
* GBM -> XGboost -> LightGBM으로 발전 History 설명
* 데이터 실습
* 고려대학교 DSBA Lab. 강필성 교수님 강의 자료를 참고함
    * Keyword : *#Missing Value Handling #Bigdata Learning #GBM #XGBoost #LightGBM #Feature Importance Score*
* Reference site
    * XGboost - Hyperparameter Tuning
        * https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    * LightGBM - Hyperparameter Tuning
        * https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
```
# Install Package
- conda install -c conda-forge xgboost
- conda install -c conda-forge lightgbm
```
### [Class09] eXplainable Method For High Complexity Models - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass10%5D%20eXplainable%20Method%20For%20High%20Complexity%20Models/C0708_eXplainable_Method_For_High_Complexity_Model.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass10%5D%20eXplainable%20Method%20For%20High%20Complexity%20Models/LightGBM_Regression_SHAP.ipynb)
* 복잡한 모델을 해석하기 위한 기법 소개
* Global Feature Importance Score VS Local Feature Importance Score
* Interpretable Meachine Learning을 통한 원인 분석
* 데이터 실습
    * Keyword : *#IML #Global VS Local #LIME #SHAP* 
* Reference site
    * Interpretable Machine Learning (IML)  
        * https://christophm.github.io/interpretable-ml-book/interpretability-importance.html
    * LIME
        * https://pythondata.com/local-interpretable-model-agnostic-explanations-lime-python/
        * https://github.com/marcotcr/lime/tree/master/lime
        * https://yjjo.tistory.com/3
    * SHAP
        * https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
        * https://datanetworkanalysis.github.io/2019/12/24/shap3
        * https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html
```
# Install Package
- conda install -c conda-forge shap
```
### [Class10] Clustering & Dimensionality Reduction - [[Slide]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass11%5D%20Clustering%20&%20Dimensionality%20Reduction/C0709_Clustering___Dimensionality_Reduction.pdf), [[Tutorial Code]](http://mod.lge.com/hub/smartdata/data-science-online-course-from-dstp/blob/master/%5BClass11%5D%20Clustering%20&%20Dimensionality%20Reduction/Clustering___Dimensonality_Reduction.ipynb)
* Unsupervised Learning을 활용하여 최적의 X's 조합을 도출함
* 복잡한 Supervised Learning을 탈피하여 고효율군을 이루는 X's들의 조합을 찾는 새로운 기법 제시
* Dimensionality Reduction을 활용하여 clsuter의 분포를 확인함
* 데이터 실습
    * Keyword : *#Distance #HDBSCAN #Spectral #PCA #T-SNE #Autoencoder* 
* Reference site
    * Clustering - 다양한 데이터에 여러 개의 Cluster 기법을 실험해 놓음
        * https://scikit-learn.org/stable/modules/clustering.html
    * HDBSCAN - 개념정리
        * https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html
```
# Install Package
- conda install -c conda-forge hdbscan
```
## Recommended Open Class
* Data Mining & Quality Analytics Lab @Korea University : Machine Learning, Deep Learning, Artifical Intelligence
    * Lab Homepage : http://dmqa.korea.ac.kr/
    * Lab Seminar : http://dmqa.korea.ac.kr/activity/seminar
    * YouTube Video : https://www.youtube.com/channel/UCueLU1pCvFlM8Y8sth7a6RQ
* Data Science & Business Analytics Lab @Korea University : Machine Learning, Deep Learning, Artifical Intelligence
    * Lab Homepage : http://dsba.korea.ac.kr/
    * Lab Seminar : http://dsba.korea.ac.kr/seminar/
    * YouTube Video : https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg
* Applied Artifical Intelligence @KAIST : Machine Learning, Deep Learning, Artifical Intelligence
    * Lab Homepage : https://aai.kaist.ac.kr/xe2/
    * YouTube Video : https://www.youtube.com/channel/UC9caTTXVw19PtY07es58NDg/videos
## Data Visualization Reference Site
* Top 50 matplotlib Visualization Code - ***초강추!!!*** 
    * https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
* 전반적인 Plotting Reference (Scatter size, form, plot 종류 등등등)
	* https://jfun.tistory.com/63
* Bubble size 조정하면서 Scatter plot 그리는 방법 Reference
    * https://medium.com/@peteryun/python-matplotlib-%EA%B8%B0%EB%B3%B8-6e23e5fd2f16
* Color bar 종류
    * https://pythonkim.tistory.com/82
* The Next Level of Data Visualization in Python
    * https://towardsdatascience.com/the-next-level-of-data-visualization-in-python-dd6e99039d5e
* BOKEH
    * http://docs.bokeh.org/en/latest/docs/user_guide/plotting.html
    * https://lovit.github.io/visualization/2018/03/31/bokeh_python_plotting/
    * https://lovit.github.io/visualization/2019/11/22/bokeh_tutorial/

