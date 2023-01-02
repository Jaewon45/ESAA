# ESAA_2022

# Description
This is self-studying records in ESAA (Ewha Statistic Analysis Association), based on 3 books and some examples, exercises, and mini-projects within a team.
The books are as below:
1. 파이썬 머신러닝 완벽 가이드 (Python Machine-Learning Definitive Guide, http://www.yes24.com/Product/Goods/108824557)
2. Do it! 데이터 분석을 위한 판다스 입문 (Pandas for Everyone: Python Data Analysis, https://www.amazon.com/Pandas-Everyone-Analysis-Addison-Wesley-Analytics-ebook/dp/B0789WKTKJ)
3. 핸즈온 머신러닝 (Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
4. 파이썬 딥러닝 텐서플로 (Python Deep-Learning Tensorflow, http://www.yes24.com/Product/Goods/102603640)


## Contents of Materials

### 1. 파이썬 머신러닝 완벽 가이드 (Python Machine-Learning Definitive Guide)

1장: 파이썬 기반의 머신러닝과 생태계 이해

2장: 사이킷런으로 시작하는 머신러닝(Sklearn for ML)   
2.1. 사이킷런 소개와 특징(Overview)     
2.2. 첫 번째 머신러닝 만들어 보기 - 붓꽃 품종 예측하기(iris dataset)   
2.3. 사이킷런의 기반 프레임워크 익히기     
2.4. Model Selection 모듈 소개    
2.5. 데이터 전처리(preprocessing)   
2.6. 사이킷런으로 수행하는 타이타닉 생존자 예측(titanic dataset)   
2.7. 정리(Summary)  

3장: 평가(Evaluation)  
3.1. 정확도(Accuracy)   
3.2. 오차 행렬(Confusion Matrix)  
3.3. 정밀도와 재현율(Precision and Recall)  
3.4. F1 스코어(F1)  
3.5. ROC 곡선과 AUC  
3.6. 피마 인디언 당뇨병 예측(Pima Indian diabetes dataset)  
3.7. 정리(Summary)  

4장: 분류(Classification)   
4.1. 분류(Classification)의 개요  
4.2. 결정 트리(Decision Tree)  
4.3. 앙상블 학습(Ensemble)  
4.4. 랜덤 포레스트(RF)  
4.5. GBM(Gradient Boosting Machine)  
4.6. XGBoost(eXtra Gradient Boost)  
4.7. LightGBM  
4.8. 분류 실습 - 캐글 산탄데르 고객 만족 예측(example 1)  
4.9. 분류 실습 - 캐글 신용카드 사기 검출(example 2)  
4.10. 스태킹 앙상블(Stacking Ensemble)  
4.11. 정리(Summary)    

5장: 회귀(Regression)   
5.1. 회귀 소개(Overview)  
5.2. 단순 선형 회귀를 통한 회귀 이해(SRL)  
5.3. 비용 최소화하기 - 경사 하강법(Gradient Descent) 소개  
5.4. 사이킷런 LinearRegression을 이용한 보스턴 주택 가격 예측(Boston Housing prices)  
5.5. 다항 회귀와 과(대)적합/과소적합 이해(Over/Underfitting)  
5.6. 규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷(Regulation-Ridge, Lasso, ElasticNet)    
5.7. 로지스틱 회귀(Logistic Regression)  
5.8. 회귀 트리(Regression Tree)    
5.9. 회귀 실습 - 자전거 대여 수요 예측(example1)  
5.10. 회귀 실습 - 캐글 주택 가격: 고급 회귀 기법(example2)  
5.11. 정리(Summary)  

6장: 차원 축소(Dimension Reduction)  
6.1. 차원 축소 개요(Overview)  
6.2. PCA(Principal Component Analysis)  
6.3. LDA(Linear Discriminant Analysis)  
6.4. SVD(Singular Value Decomposition)  
6.5. NMF(Non-Negative Matrix Factorization)  
6.6. 정리(Summary)   

7장: 군집화(Clustering)  
7.1. K-평균 알고리즘 이해(K-means Algorithms)  
7.2. 군집 평가(Cluster Evaluation)  
7.3. 평균 이동    
7.4. GMM(Gaussian Mixture Model)  
7.5. DBSCAN  
7.6. 군집화 실습 - 고객 세그먼테이션(Example-Customer Segmentation)    
7.7. 정리(Summary)  

8장: 텍스트 분석(Text Analysis)  
8.1. 텍스트 분석 이해(Overview)  
8.2. 텍스트 사전 준비 작업(텍스트 전처리) - 텍스트 정규화(Text Normalization)  
8.3. Bag of Words - BOW  
8.4. 텍스트 분류 실습 - 20 뉴스그룹 분류(20 News Group dataset)  
8.5. 감성 분석(Sentiment Analysis)    
8.6. 토픽 모델링(Topic Modeling) - 20 뉴스그룹(20 News Group dataset)  
8.7. 문서 군집화 소개와 실습(Opinion Review 데이터 세트)  
8.8. 문서 유사도(Text Similarity)     
8.9. 한글 텍스트 처리 - 네이버 영화 평점 감성 분석(Korean Text Analysis)  
8.10. 텍스트 분석 실습-캐글 Mercari Price Suggestion Challenge   
8.11. 정리(Summary)  

9장: 추천 시스템(Recommendation)  
9.1. 추천 시스템의 개요와 배경(Overview)  
9.2. 콘텐츠 기반 필터링 추천 시스템(Contents-based Filtering)  
9.3. 최근접 이웃 협업 필터링(Nearest Neighbor Collaborative Filtering)  
9.4. 잠재 요인 협업 필터링(Latent Factor Collaborative Filtering)  
9.5. 콘텐츠 기반 필터링 실습 - TMDB 5000 영화 데이터 세트   
9.6. 아이템 기반 최근접 이웃 협업 필터링 실습(Item-based Nearest Neighbor Collaborative Filtering)   
9.7. 행렬 분해를 이용한 잠재 요인 협업 필터링 실습   
9.8. 파이썬 추천 시스템 패키지 - Surprise Package   
9.9. 정리(Summary)  



### 2. Do it! 데이터 분석을 위한 판다스 입문 (Pandas for Everyone: Python Data Analysis)


Chapter 3. Plotting Basics   
3.1 Why Visualize Data?       
3.2 Matplotlib Basics      
3.3 Statistical Graphics Using matplotlib      
3.4 Seaborn      
3.5 Pandas Plotting Method      

Chapter 4. Tidy Data      
4.1 Columns Contain Values, Not Variables      
4.2 Columns Contain Multiple Variables      
4.3 Variables in Both Rows and Columns      

Chapter 11. Strings and Text Data     
11.1 Strings       
11.2 String Methods      
11.3 More String Methods      
11.4 String Formatting (F-Strings)       
11.5 Regular Expressions (RegEx)      
11.6 The regex Library      

Chapter 12. Dates and Times      
12.1 Python's datetime Object      
12.2 Converting to datetime      
12.3 Loading Data That Include Dates      
12.4 Extracting Date Components      
12.5 Date Calculations and Timedeltas      
12.6 Datetime Methods      
12.7 Getting Stock Data         
12.8 Subsetting Data Based on Dates    
12.9 Date Ranges      
12.10 Shifting Values      
12.11 Resampling      
12.12 Time Zones       
12.13 Arrow for Better Dates and Times      


### 3. 핸즈온 머신러닝 (Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)

I. The Fundamentals of Machine Learning
Ch 1. The Machine Learning Landscape  
Ch 2. End-to-End Machine Learning Project  
Ch 3. Classification  
Ch 4. Training Models  
Ch 5. Support Vector Machines    
Ch 6. Decision Trees  
Ch 7. Ensemble Learning and Random Forests  
Ch 8. Dimensionality Reduction  
Ch 9. Unsupervised Learning Techniques  

### 4. 파이썬 딥러닝 텐서플로 (Python Deep-Learning Tensorflow)

PART 03 케라스(Keras)  
01 딥러닝 준비  
02 단순 신경망 훈련  
03 심층 신경망으로 이미지 분류  
04 모델 세부 설정  
05 콜백(Callback)  
06 모델 저장 및 불러오기  
07 복잡한 모델 생성  
08 사용자 정의  
09 텐서플로 데이터셋  
10 tf.data.Dataset 클래스  

PART 04 합성곱 신경망(CNN)  
01 합성곱 신경망  
02 간단한 모델 생성  
03 복잡한 모델 생성  
04 위성 이미지 분류  
05 개/고양이 분류  
06 객체 탐지(Object Detection)  
07 이미지 분할(Segmentation)  

