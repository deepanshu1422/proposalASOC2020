# Alibaba Summer of Code Proposal

  

## Part 0: Table of Contents

  

-  [Part 1: Biographical Information](#part-1--biographical-information)

*  [1.1 Basics](#11-basics)

*  [1.2 Personal Experience](#12-personal-experience)

-  [Part 2: Traffic Forecasting for Flink Job](#part-2--traffic-forecasting-for-flink-job)

*  [2.1 Background](#21-background)

+  [2.1.1 Flink](#211-flink)

+  [2.1.2 Time Series Prediction](#212-time-series-prediction)

*  [2.2 Problem Description](#22-problem-description)

*  [2.3 Time Series Forecasting Algorithms](#23-time-series-forecasting-methods)

*  [2.4 Determinants](#24-determinants)

+  [2.4.1 Data Characteristics](#241-data-characteristics-and-key-metrics)

+  [2.4.2 Adjusting Arima Performance](#242-adjusting-arima-performance)

+  [2.4.3 Analysis ](#243-analysis)

+  [2.4.4 Other](#244-other)

*  [2.5 Strategy](#25-strategy)

+  [2.5.2 Style Selection](#252-brief-overviews)

*  [2.6 Strategy Evaluation](#26-strategy-evaluation)

+  [2.6.1 Evaluating model](#261-layout-evaluation)

*  [2.7 Further Ideas](#27-further-ideas)

+  [2.7.1 An interactive Dashboard](#271-layout-transformation)

*  [2.8 References](#28-references)

-  [Part 3: Project Plan](#part-3--project-plan)

*  [3.1 Timeline](#31-timeline)

*  [3.2 Deliverables](#32-deliverables)

-  [Part 4: Other Information](#part-4--other-information)

  
  

## Part 1: Biographical Information

  

### 1.1 Basics

  

- Name: Deepanshu Udhwani

- GitHub: [deepanshu1422](https://github.com/deepanshu1422).

- Email: deepanshu1422@gmail.com.

- Telephone: +91-7018765080.

- Major: Computer Science And Engineering At Thapar Institute Of Engineering And Technology.

- Skills: Python developer; Java/C++ for Data Structures, Scala for mathematic computation, MATLAB.

- CV: [Curriculum vitae]([https://drive.google.com/file/d/17OTQqXDRsaSmJURa6Iy_l2lGvSRiUFzC/view](https://drive.google.com/file/d/17OTQqXDRsaSmJURa6Iy_l2lGvSRiUFzC/view)).

  

### 1.2 Personal Experience

  
Being a computer science student with a specialization in Electronics and Communication. I am currently enrolled in as MBA student with majors in Information Systems and Marketing I am interested in kernels and machine learning from my freshman year, I think this is a very good opportunity for me to get hands-on development experience in this field. I am prepared to learn more about it in the summers and explore this interesting field.

  

I previously interned at the Computation Acceleration team at Thapar Institute Of Engineering And Technology wherein I developed a synthesizable arbitrary precision fixed and floating-point library for their High-Level Synthesis tool which also works with Vivado and Calypto and also wrote the complete documentation and end user manuals. I thus feel that I have experience with the concepts and skills needed to complete this project. Alibaba Summer of Coding provides me with a chance to make contributions to open source projects, with mentorship from great developers from Alibaba. I cherish this opportunity and would put my 100% to devote to this project!

  
  

## Part 2: ****Traffic Forecasting for Flink Job****

  

### 2.1 Background

  

#### 2.1.1 Flink

 Apache Flink is a framework and distributed processing engine for stateful computations over _unbounded and bounded_ data streams
It provides multiple APIs at different levels of abstraction and offers dedicated libraries for common use cases 
  

#### 2.1.2 Time Series Prediction

  

We all know that time series prediction is an important part of machine learning that could save us a lot of time in prediction of load, services, servers and many other factors that will help us in long run.
This often involves us making assumptions about the form of data and decomposing the time series into constitution components

  

### 2.2 Problem Description

  By analysing data we found out that most of the fink jobs are streaming jobs that are long learning and varies smoothly by time. We here need to develop a model to support traffic forecasting for the Flink job so that we can adjust the configuration of job dynamically according to the traffic need to process 


### 2.3 Time Series Forecasting  Algorithms

  Let us take a look at few of different time series forecasting algorithms that are widely used.

* Naive Method
* Simple Average
* Moving Average 
* Simple Exponential Smoothing
* Holt's Linear Trend
* Holt's Winter
* ARIMA

![Root Mean Square Error](https://i.ibb.co/yh4w3Hx/Annotation-2020-06-26-154651.png)


Through the analysis of Root Mean Square Errors of these models we can suggest that we should go with either **Holt's Winter** or **ARIMA** model
  
But as ARIMA has more perimeters in its arguments I would go ahead with **AutoRegressive Integrated Moving Average**   or with **SARIMA** as it might be able to give better results 

### 2.4 Determinants

  

#### 2.4.1 Data Characteristics and Key Metrics

  

Data characteristics provide the most important clues for us to determine which model to use. the configuration parameters of various models have shown some great differences. ARIMA can automatically predict the output and trend  based on data characteristics when forecasting the next event

  

**Table 1. is a brief analysis of the key metrics that our model needs to take care while processing**

  1. General Health
  ![Key Metrics ](https://i.ibb.co/y522mnr/Annotation-2020-06-26-161755.png)

 2. Throughput 
 ![enter image description here](https://i.ibb.co/v37Mz7z/Annotation-2020-06-26-162301.png)

3. Monitoring Latency
![enter image description here](https://i.ibb.co/qjYXxPs/Annotation-2020-06-26-162548.png)

4. **Accuracy Metrics for our model will be determined by these parameters**

*   Mean Absolute Percentage Error (MAPE)
*  Mean Error (ME)
*  Mean Absolute Error (MAE)
*  Mean Percentage Error (MPE)
*  Root Mean Squared Error (RMSE)
*  Lag 1 Autocorrelation of Error (ACF1)
*  Correlation between the Actual and the Forecast (corr)
*  Min-Max Error (min-max)

#### 2.4.2 Adjusting ARIMA Model Performance
  
 There are many parameters to consider while configuring ARIMA model but  we are majorly concerned with these three things:

1. How to suppress noisy output when fitting an ARIMA model
2. Effect of Enabling or Disabling trend term in our model
3. Adjusting coefficients during the training of our data
  
Given most of the jobs will be long-running and traffic will be varying smoothly by time. We'd be able to adjust the parameters dynamically according to the traffic  

#### 2.4.3 Analysis 
The model must be able to achieve these things for Apache Flink correctly and we are done :)
-   Ability to support an unbounded number of customer properties 
-   Low latency
-   Fault-tolerant
- Plotting real-time graph and adjusting model to predict the best

  

### 2.5 Strategy

* Visualizing the Time Series Data
* Preprocessing: creating time stamps, making series univariate etc.
* Making series stationery for that moment
* Determining difference value
* Creating ACF and PACF the most important step that is used to determine the input parameters 

 **Creating model** 
  

    from statsmodels.tsa.statespace.sarimax import SARIMAX 
    model = SARIMAX(df, order =(p,d,q))
    
**Fitting model** 

    model.fit()

  **Making Forecast**
  
    mean_forecast = results.get_forecast(steps=10).predicted_mean

![ARIMA Model Workflow](https://www.researchgate.net/publication/336544972/figure/fig1/AS:814079476240385@1571103080961/ARIMA-model-flow-chart.ppm)

  **Calculating RMSE and adjusting the model accordingly**
  

    rmse(data,forecast)

#### 2.5.2 Brief Overviews



##### 2.5.2.2 Brief overview components

1.Flink Architecture
![enter image description here](https://i.ibb.co/WG1D2YG/Annotation-2020-06-26-200615.png )


2. Streaming Analytics

  ![enter image description here](https://i.ibb.co/xHFgGtx/Annotation-2020-06-26-200314.png)


3.Multi Inputs in a Process

![enter image description here](https://flink.apache.org/img/blog/blog_multi_input.png)


4. Load Balancing using ARIMA Model


![ARIMA Workload Analyzer](https://i.ibb.co/kXzNcv1/Annotation-2020-06-27-110002.png)

5.Evaluating The Forecast


![enter image description here](https://docs.aws.amazon.com/forecast/latest/dg/images/evaluation-offset.png)

6. Evaluating Backtests
  
  ![enter image description here](https://docs.aws.amazon.com/forecast/latest/dg/images/evaluation-backtests.png)

  

### 2.6 Strategy Evaluation

  

  

#### 2.6.1  Evaluating ARIMA model

  


    # evaluating an ARIMA model for a given order (p,d,q)
    
    def evaluate_arima_model(X,  arima_order):
    
    # prepare training dataset
    
    train_size  =  int(len(X)  *  0.66)
    
    train,  test  =  X[0:train_size],  X[train_size:]
    
    history  =  [x  for  x  in  train]
    
    # make predictions
    
    predictions  =  list()
    
    for  t  in  range(len(test)):
    
    model  =  ARIMA(history,  order=arima_order)
    
    model_fit  =  model.fit(disp=0)
    
    yhat  =  model_fit.forecast()[0]
    
    predictions.append(yhat)
    
    history.append(test[t])
    
    # calculate out of sample error
    
    error  =  mean_squared_error(test,  predictions)
    
    return  error

  

### 2.7 Further Ideas

  

#### 2.7.1 An interactive Dashboard 

* The Dashboard will give the ability to visualise the background processing.
* Ability to fine-tune the model by yourself.
* Better user experience while using Apache-FLINK metrics

  



### 2.8 References

  

[1] ARIMA documentation. [https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

  

[2] Apache Flink documentation. [https://ci.apache.org/projects/flink/flink-docs-stable/](https://ci.apache.org/projects/flink/flink-docs-stable/)

  

[3] Wikipedia Time Series. 
[https://en.wikipedia.org/wiki/Time_series](https://en.wikipedia.org/wiki/Time_series)

  

[4] Siraj Raval Time-series Prediction 
https://www.youtube.com/watch?v=d4Sn6ny_5LI

  

[5] Data Science Dojo Time Series 
https://www.youtube.com/watch?v=wGUV_XqchbE

  

[6] How Flink works. 
https://flink.apache.org/flink-architecture.html#:~:text=Apache%20Flink%20is%20a%20framework,speed%20and%20at%20any%20scale.

  

[7] Third Party projects for Flink
 https://flink-packages.org/

  

[8] Flink Streaming Example [https://flink.apache.org/news/2015/02/09/streaming-example.html](https://flink.apache.org/news/2015/02/09/streaming-example.html)

  

[9] Flink Serialization.
 [https://flink.apache.org/news/2020/04/15/flink-serialization-tuning-vol-1.html](https://flink.apache.org/news/2020/04/15/flink-serialization-tuning-vol-1.html)

  

[10] How ARIMA Works [https://medium.com/fintechexplained/understanding-auto-regressive-model-arima-4bd463b7a1bb](https://medium.com/fintechexplained/understanding-auto-regressive-model-arima-4bd463b7a1bb)

  

[11] Different Time Series Models.
[https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/) 
  

[12] AWS Forecast
[https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-arima.html](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-arima.html)
  

[13] Predictor Metrics
https://docs.aws.amazon.com/forecast/latest/dg/images/evaluation-backtests.png

  

[14] Stack Overflow ARIMA
[https://stackoverflow.com/questions/53629933/forecasting-with-arima](https://stackoverflow.com/questions/53629933/forecasting-with-arima)

  

[15] Dev.to articles. [https://dev.to/paveltiunov/time-series-anomaly-detection-algorithms-4gmj](https://dev.to/paveltiunov/time-series-anomaly-detection-algorithms-4gmj)

  


## Part 3: Project Plan

  

### 3.1 Timeline

  

I live in New Delhi, India, working in UTC+5:30 timezone. I believe it would take about 7\~8 weeks for me to complete the project. Before student projects announced in early July, I could have a head start with some early preparation. I have plenty of time to complete this project during this period and it would be nice for me to work 40\~50 hours a week. This is my detailed timeline.

  

| Date | &nbsp;&nbsp;&nbsp;Progress |

| -------------- | -------------------- |

| July 7 \~ 9  &nbsp;&nbsp;&nbsp;| **Community Bonding Period**<br />- Figure out the features and logics inside our project.<br />- Discussion with mentors about the new features.<br />- Get familiar with community rules (on GitHub, Gitter, User Group). |

| July 9 \~ 16 | &nbsp;&nbsp;&nbsp;**Compare and contrast different model parameters**<br />- Surveying data characteristics that model will use.<br />- Designing an appropriate  path and solution in accordance with mentor. |

| July 17 \~ 24 | &nbsp;&nbsp;&nbsp;**Creating and Feeding AIMA**<br />- Creating and feeding data to ARIMA model.<br />- Training online on per job basis since different jobs might have different patterns of traffic. |

| July 25 \~ 31 | &nbsp;&nbsp;&nbsp;**Running model on static historical  data**<br />- Analysis of Results.<br />- Design an appropriate interface for feeding and extracting data from the model.|

| August 1\~ 4 | &nbsp;&nbsp;&nbsp;**Mid-term Evaluation**<br />- Provide examples and suitable reference of documentation.<br />- Discuss with mentors about the next steps. |

| August 5 \~ 15 | &nbsp;&nbsp;&nbsp;**Adding support for dynamic model parameters**<br />- Refining feature to make model make dynamic calculations .<br />- Streamlining the process by reducing the gap between observed rate and predicted rate. |

| August 16 \~ 23 | &nbsp;&nbsp;&nbsp;**Training and Testing/Documentation**<br />- Testing.<br />- Writing documentation and refactoring the codebase |

| August 24 \~ 27 | &nbsp;&nbsp;&nbsp;**Evaluate Traffic Forecasting for Flink Job**<br />- Evaluate the model and results.<br />- Final Tweaking |

| August 28 \~ 31 | &nbsp;&nbsp;&nbsp;**Final Evaluation**<br />- Provide examples and API reference of documentation.<br />- Write a summary article throughout the project. |

  

### 3.2 Deliverables

  

I'll be implementing a time series forecasting model for Flink jobs. The project can be roughly divided into three constitutes: The model selection is done as ARIMA as it performs better than other models for this particular use case; The model will be able to adjust dynamically according to the nature of Flink jobs and do the necessary things to reduce cost and optimise the computation; The evaluation use the feedbacks and criteria of the smooth functioning and correctness in predictability to validate the effectiveness of our system. This project will play a significant role in Flink ecosystem. This project will provide users with analysis and help fink to deliver more good with modest costs.

  

## Part 4: Other Information

  

This is my first time participating ASoC and I love working with Machine Learning and Kernel Development projects. I would love to work on this task and will give my 100% for the timely completion of this project  
